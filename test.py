import torch
import torchaudio

from audio_tokenizer.modeling_audio_vae import AudioVAE

model = AudioVAE.from_pretrained('inclusionAI/Ming-omni-tts-tokenizer-12Hz')
model = model.cuda()
model.eval()
model = model.to(torch.bfloat16)

sample_rate = model.config.sample_rate
waveform, sr = torchaudio.load('data/1089-134686-0000.flac', backend='soundfile')
if sr != sample_rate:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
sample = {'waveform': waveform.cuda(), 'waveform_length': torch.tensor([waveform.size(-1)]).cuda()}

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent, frame_num = model.encode_latent(**sample)
        output_waveform = model.decode(latent)

torchaudio.save('./1089-134686-0000_reconstruct.wav', output_waveform.cpu()[0], sample_rate=sample_rate)
