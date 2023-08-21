import os
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
model.to(device)

sample_rate = bundle.sample_rate

def separate_sources(
    model,
    mix,
    segment=10.,
    overlap=0.1,
    device=None,
):
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = T.Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape='linear')

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def separate_and_save_sources(mp3_path, output_dir):
    waveform, sr = torchaudio.load(mp3_path)
    waveform = waveform.to(device)
    mixture = waveform

    segment: int = 10
    overlap = 0.1

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
    )[0]
    sources = sources * ref.std() + ref.mean()

    file_name = os.path.splitext(os.path.basename(mp3_path))[0]
    output_dir = os.path.join(output_dir, file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, source in enumerate(model.sources):
        output_path = os.path.join(output_dir, f"{source}.wav")
        torchaudio.save(output_path, sources[i], sample_rate)

def process_all_mp3_in_folder(input_folder_path, output_folder_path):
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(input_folder_path, filename)
            separate_and_save_sources(mp3_path, output_folder_path)

# Call the function with the paths to the input and output folders you want to process
# process_all_mp3_in_folder('your_input_folder_path', 'your_output_folder_path')
audio_samples = './audio_sample'
output_folder = './separate_sample'

process_all_mp3_in_folder(audio_samples, output_folder)