import torch
import torchaudio

from audio_tokenizer.modeling_audio_vae import AudioVAE

model = AudioVAE.from_pretrained('inclusionAI/Ming-UniAudio-Tokenizer')
model = model.cuda()
model.eval()

waveform, sr = torchaudio.load('data/1089-134686-0000.flac', backend='soundfile')
sample = {'waveform': waveform.cuda(), 'waveform_length': torch.tensor([waveform.size(-1)]).cuda()}

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent, frame_num = model.encode_latent(**sample)
        output_waveform = model.decode(latent)

torchaudio.save('./1089-134686-0000_reconstruct.wav', output_waveform.cpu()[0], sample_rate=16000)