<p align="center">
    <img src="./assets/ant-bailing.png" width="100"/>
<p>

<p align="center">📑 <a href="">Technical Report</a>｜📖<a href="https://xqacmer.github.io/Ming-Unitok-Audio.github.io">Project Page</a> ｜🤗 <a href="https://huggingface.co/inclusionAI/MingTok-Audio">Hugging Face</a>｜ 🤖 <a href="https://modelscope.cn/models/inclusionAI/MingTok-Audio">ModelScope</a>

## Architecture
<!-- ![MingTok-Audio](assets/uniaudio-tokenizer.png)
![MingTok-Audio-training](assets/uniaudio-tokenizer-training.png) -->

<p align="center">
  <img src="assets/uniaudio-tokenizer.png" alt="MingTok-Audio"/>
</p>
<p align="center">
  <img src="assets/uniaudio-tokenizer-training.png" alt="MingTok-Audio-training"/>
</p>

## Key Features
- 🚀 **First Unified Continuous Speech Tokenizer:** the first continuous audio tokenizer to effectively integrate semantic and acoustic features, suitable for both understanding and generation tasks.
- 🎧 **High-Fidelity Reconstruction:** Achieve high-fidelity audio generation by modeling continuous features with a VAE, minimizing information loss and preserving intricate acoustic textures.
- 🌐 **Convolution-Free Efficiency:** Built on a pure causal transformer architecture, completely eliminating convolutional layers for superior efficiency and a simpler design.



## Installation
```
pip install -r requirements.txt
```

## Quick start
```python
import torch
import torchaudio

from audio_tokenizer.modeling_audio_vae import AudioVAE

model = AudioVAE.from_pretrained('inclusionAI/MingTok-Audio')
model = model.cuda()
model.eval()

waveform, sr = torchaudio.load('data/1089-134686-0000.flac', backend='soundfile')
sample = {'waveform': waveform.cuda(), 'waveform_length': torch.tensor([waveform.size(-1)]).cuda()}

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent, frame_num = model.encode_latent(**sample)
        output_waveform = model.decode(latent)

torchaudio.save('./1089-134686-0000_reconstruct.wav', output_waveform.cpu()[0], sample_rate=16000)
```

## Performance
### Speech reconstruction performance
<table>
  <caption>Speech reconstruction performance comparison on various audio benchmark datasets. The best results are in <strong>bold</strong>.</caption>
  <thead>
    <tr>
      <th rowspan="2" align="left"><b>System</b></th>
      <th rowspan="2" align="center"><b>FrameRate</b></th>
      <th colspan="3" align="center"><b>SEED-ZH</b></th>
      <th colspan="3" align="center"><b>SEED-EN</b></th>
    </tr>
    <tr>
      <th align="center"><b>PESQ↑</b></th>
      <th align="center"><b>SIM↑</b></th>
      <th align="center"><b>STOI↑</b></th>
      <th align="center"><b>PESQ↑</b></th>
      <th align="center"><b>SIM↑</b></th>
      <th align="center"><b>STOI↑</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">MiMo-Audio-Tokenizer</td>
      <td align="center">25</td>
      <td align="center">2.71</td>
      <td align="center">0.89</td>
      <td align="center">0.93</td>
      <td align="center">2.43</td>
      <td align="center">0.85</td>
      <td align="center">0.92</td>
    </tr>
    <tr>
      <td align="left">GLM4-Voice-Tokenizer</td>
      <td align="center">12.5</td>
      <td align="center">1.06</td>
      <td align="center">0.33</td>
      <td align="center">0.61</td>
      <td align="center">1.05</td>
      <td align="center">0.12</td>
      <td align="center">0.60</td>
    </tr>
    <tr>
      <td align="left">Baichuan-Audio-Tokenizer</td>
      <td align="center">12.5</td>
      <td align="center">1.84</td>
      <td align="center">0.78</td>
      <td align="center">0.86</td>
      <td align="center">1.62</td>
      <td align="center">0.69</td>
      <td align="center">0.85</td>
    </tr>
    <tr>
      <td align="left">XY-Tokenizer</td>
      <td align="center">12.5</td>
      <td align="center">2.27</td>
      <td align="center">0.77</td>
      <td align="center">0.90</td>
      <td align="center">2.14</td>
      <td align="center">0.82</td>
      <td align="center">0.90</td>
    </tr>
    <tr>
      <td align="left">Mimi</td>
      <td align="center">75</td>
      <td align="center">2.05</td>
      <td align="center">0.73</td>
      <td align="center">0.89</td>
      <td align="center">2.01</td>
      <td align="center">0.77</td>
      <td align="center">0.89</td>
    </tr>
    <tr>
      <td align="left">XCodec2.0</td>
      <td align="center">50</td>
      <td align="center">2.19</td>
      <td align="center">0.80</td>
      <td align="center">0.92</td>
      <td align="center">2.37</td>
      <td align="center">0.82</td>
      <td align="center">0.93</td>
    </tr>
    <tr>
      <td align="left">BigCodec</td>
      <td align="center">80</td>
      <td align="center">2.26</td>
      <td align="center">0.81</td>
      <td align="center">0.92</td>
      <td align="center">2.22</td>
      <td align="center">0.80</td>
      <td align="center">0.91</td>
    </tr>
    <tr>
      <td align="left"><strong>MingTok-Audio(ours)</td>
      <td align="center">50</td>
      <td align="center"><b>4.21</b></td>
      <td align="center"><b>0.96</b></td>
      <td align="center"><b>0.98</b></td>
      <td align="center"><b>4.04</b></td>
      <td align="center"><b>0.96</b></td>
      <td align="center"><b>0.98</b></td>
    </tr>
  </tbody>
</table>
 

### The adaptation performance for downstream ASR tasks
<table>
  <caption>Understanding ASR performance comparison on various audio benchmark datasets. The best results are in <strong>bold</strong>.</caption>
  <thead>
    <tr>
      <th rowspan="2"><strong>Datasets</strong></th>
      <th rowspan="2"><strong>Model</strong></th>
      <th colspan="7"><strong>Performance</strong></th>
    </tr>
    <tr>
      <th><strong>aishell2-ios</strong></th>
      <th><strong>LS-clean</strong></th>
      <th><strong>Hunan</strong></th>
      <th><strong>Minnan</strong></th>
      <th><strong>Guangyue</strong></th>
      <th><strong>Chuanyu</strong></th>
      <th><strong>Shanghai</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><strong>Understanding ASR</strong></td>
      <td>Kimi-Audio</td>
      <td><strong>2.56</td>
      <td><strong>1.28</td>
      <td>31.93</td>
      <td>80.28</td>
      <td>41.49</td>
      <td>6.69</td>
      <td>60.64</td>
    </tr>
    <tr>
      <td>Qwen2.5 Omni</td>
      <td>2.75</td>
      <td>1.80</td>
      <td>29.31</td>
      <td>53.43</td>
      <td>10.39</td>
      <td>7.61</td>
      <td>32.05</td>
    </tr>
    <tr>
      <td>Qwen2 Audio</td>
      <td>2.92</td>
      <td>1.60</td>
      <td>25.88</td>
      <td>123.78</td>
      <td>7.59</td>
      <td>7.77</td>
      <td>31.73</td>
    </tr>
    <tr>
      <td><strong>Ming-UniAudio-16A3B(ours)</td>
      <td>2.84</td>
      <td>1.62</td>
      <td><strong>9.80</strong></td>
      <td><strong>16.50</strong></td>
      <td><strong>5.51</strong></td>
      <td><strong>5.46</strong></td>
      <td><strong>14.65</strong></td>
    </tr>
  </tbody>
</table> 

### The adaptation performance for downstream TTS tasks
  
<table>
<caption>Performance comparison on various audio benchmark datasets. The best results are in <strong>bold</strong>.</caption>
  <thead>
    <tr>
      <th align="left"><b>Datasets</b></th>
      <th align="left"><b>Model</b></th>
      <th colspan="4" align="center"><b>Performance</b></th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th align="center"><b>Seed-zh WER(%)</b></th>
      <th align="center"><b>Seed-zh SIM</b></th>
      <th align="center"><b>Seed-en WER(%)</b></th>
      <th align="center"><b>Seed-en SIM</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" align="left" style="vertical-align: middle;"><b>Generation</b></td>
      <td align="left">Seed-TTS</td>
      <td align="center">1.12</td>
      <td align="center"><b>0.80</b></td>
      <td align="center">2.25</td>
      <td align="center"><b>0.76</b></td>
    </tr>
    <tr>
      <td align="left">MiMo-Audio</td>
      <td align="center">1.96</td>
      <td align="center">-</td>
      <td align="center">5.37</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left">Qwen3-Omni-30B-A3B-Instruct</td>
      <td align="center">1.07</td>
      <td align="center">-</td>
      <td align="center"><b>1.39</b></td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left">Ming-Omni-Lite</td>
      <td align="center">1.69</td>
      <td align="center">0.68</td>
      <td align="center">4.31</td>
      <td align="center">0.51</td>
    </tr>
    <tr>
      <td align="left"><strong>Ming-UniAudio-16A3B(ours)</td>
      <td align="center"><b>0.95</b></td>
      <td align="center">0.70</td>
      <td align="center">1.85</td>
      <td align="center">0.58</td>
    </tr>
  </tbody>
</table>


## Acknowledgements
1. We borrowed a lot of code from [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0.git) for tokenizer training.
2. We thank the OpenAI team for developing the [Whisper](https://github.com/openai/whisper) model and making its weights publicly available.


## License and Legal Disclaimer

This code repository is licensed under the [MIT License](./LICENSE), and the Legal Disclaimer is located in the [LEGAL.md file](./LEGAL.md) under the project's root directory.

## Citation

If you find our work helpful, feel free to give us a cite.
