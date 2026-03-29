import sys
import os
import argparse
import numpy as np  # Add this with other imports
# Initialize model root path first
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model_root", type=str, default="~/Desktop/models")
args, _ = parser.parse_known_args()
MODEL_ROOT = os.path.expanduser(args.model_root)

# Add both possible hifigan paths to Python path
sys.path.append("hifigan")  # Original location
sys.path.append(os.path.join(MODEL_ROOT, "hifigan"))  # New location

# Rest of imports
import torch
from espnet2.bin.tts_inference import Text2Speech
from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from env import AttrDict
import json
import yaml
from text_preprocess_for_inference import TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor

SAMPLING_RATE = 22050

def load_hifigan_vocoder(language, gender, device):
    """Load HiFi-GAN vocoder for specified gender"""
    vocoder_config = os.path.join(MODEL_ROOT, f"vocoder/{gender}/aryan/hifigan/config.json")
    vocoder_generator = os.path.join(MODEL_ROOT, f"vocoder/{gender}/aryan/hifigan/generator")
    
    # Verify vocoder files exist
    if not all(os.path.exists(p) for p in [vocoder_config, vocoder_generator]):
        raise FileNotFoundError(f"Vocoder files not found in {os.path.dirname(vocoder_config)}")

    with open(vocoder_config, 'r') as f:
        h = AttrDict(json.loads(f.read()))
    
    torch.manual_seed(h.seed)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_fastspeech2_model(language, gender, device):
    """Load FastSpeech2 model for specified language and gender"""
    config_path = os.path.join(MODEL_ROOT, f"{language}/{gender}/model/config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Update stats file paths
    stats_files = {
        "normalize_conf": "feats_stats.npz",
        "pitch_normalize_conf": "pitch_stats.npz", 
        "energy_normalize_conf": "energy_stats.npz"
    }
    
    for conf_key, filename in stats_files.items():
        stats_path = os.path.join(MODEL_ROOT, language, gender, "model", filename)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        config[conf_key]["stats_file"] = stats_path
        
    # Save updated config
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    
    # Load model
    tts_model = os.path.join(MODEL_ROOT, f"{language}/{gender}/model/model.pth")
    if not os.path.exists(tts_model):
        raise FileNotFoundError(f"Model file not found: {tts_model}")
    
    return Text2Speech(train_config=config_path, model_file=tts_model, device=device)

def text_synthesis(language, gender, sample_text, vocoder, MAX_WAV_VALUE, device, alpha):
    """Perform end-to-end text-to-speech synthesis"""
    with torch.no_grad():
        model = load_fastspeech2_model(language, gender, device)
        out = model(sample_text, decode_conf={"alpha": alpha})
        print("TTS Done")
        
        x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
        x = x.to(device)
        
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        return audio.cpu().numpy().astype('int16')

def split_into_chunks(text, words_per_chunk=100):
    """Split long text into manageable chunks"""
    words = text.split()
    return [' '.join(words[i:i + words_per_chunk]) 
            for i in range(0, len(words), words_per_chunk)]

if __name__ == "__main__":
    # Main argument parser
    main_parser = argparse.ArgumentParser(parents=[parser], description="Text-to-Speech Inference")
    main_parser.add_argument("--language", type=str, required=True, 
                           help="Language (e.g., hindi, english, urdu)")
    main_parser.add_argument("--gender", type=str, required=True,
                           help="Gender (e.g., male, female)")
    main_parser.add_argument("--sample_text", type=str, required=True,
                           help="Text to be synthesized")
    main_parser.add_argument("--output_file", type=str, default="",
                           help="Output WAV file path")
    main_parser.add_argument("--alpha", type=float, default=1.0,
                           help="Speech rate control (1.0=normal, >1.0=slower)")
    args = main_parser.parse_args()

    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocoder = load_hifigan_vocoder(args.language, args.gender, device)
    
    # Select appropriate preprocessor
    if args.language == "urdu" or args.language == "punjabi":
        preprocessor = CharTextPreprocessor(dict_location=os.path.join(MODEL_ROOT, "phone_dict"))
    elif args.language == "english":
        preprocessor = TTSPreprocessor(dict_location=os.path.join(MODEL_ROOT, "phone_dict"))
    else:
        preprocessor = TTSDurAlignPreprocessor(dict_location=os.path.join(MODEL_ROOT, "phone_dict"))

    # Process text and synthesize
    audio_arr = []
    for text_chunk in split_into_chunks(args.sample_text):
        preprocessed_text, _ = preprocessor.preprocess(
            text_chunk, args.language, args.gender, {})
        audio = text_synthesis(
            args.language, args.gender, 
            " ".join(preprocessed_text), 
            vocoder, MAX_WAV_VALUE, device, args.alpha
        )
        audio_arr.append(audio)

    # Save output
    output_file = args.output_file or f"{args.language}_{args.gender}_output.wav"
    write(output_file, SAMPLING_RATE, np.concatenate(audio_arr))
    print(f"Audio saved to {output_file}")