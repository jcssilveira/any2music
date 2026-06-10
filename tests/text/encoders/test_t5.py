import torch

from any2music.text.encoders import T5Conditioner
from any2music.audio.decoders.musicgen import MUSICGEN_SIZES

AUDIO_PATH = "./samples/description/zelda.txt"
TEST_SECs = 15
UPDATES = 100

def test_t5_encode():
    dec_size = MUSICGEN_SIZES["test"]
    t5 = T5Conditioner('t5-base', dec_size.d_model, device='cuda').cuda()

    text = ""
    with open(AUDIO_PATH, 'r') as f:
        text = f.read()
    print(f"text: {text}\n")

    t5_inputs = t5.tokenize([text])
    print(f"t5_inputs shape-> {t5_inputs['input_ids'].shape}")
    print(f"t5_inputs -> {t5_inputs['input_ids']}\n")

    with torch.no_grad():
        t5_embeds, _ = t5(t5_inputs)
        print(f"t5_embeds shape -> {t5_embeds.shape}")
        print(f"t5_embeds -> {t5_embeds}")

