import argparse
import gradio as gr
from gradio import components
import os
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
from scipy.signal import resample
from scipy.io import wavfile
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,SynthesizerTrnMs768NSFsid,SynthesizerTrnMs768NSFsid_nono
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
import glob
device = "cuda:0"  # or cpu
is_half = True  # NVIDIA 20 series and higher GPUs are half-precise with no change in quality
global_vcid=None

vc_models = ['No conversion']
# load rvc model names
os.makedirs('rvc/', exist_ok=True)
vc_model_root = 'rvc'
vc_models.extend([d for d in os.listdir(vc_model_root) if os.path.isdir(os.path.join(vc_model_root, d))])


def get_vc(sid):
    weight_root = "rvc"
    voicemodel = glob.glob(f"{weight_root}/{sid}/*.pth")[0]
    global cpt,tgt_sr,version
    print("loading %s" % voicemodel)
    cpt = torch.load(voicemodel, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, device, is_half)
    n_spk = cpt["config"][-3]
    return vc, net_g

'''

def get_vc(sid):
    weight_root = "rvc"
    voicemodel = glob.glob(f"{weight_root}/{sid}/*.pth")[0]
    global cpt

    cpt = torch.load(voicemodel, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if (if_f0 == 1):
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device)

    if (is_half):
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, device, is_half)
    n_spk = cpt["config"][-3]

    return vc, net_g
'''
# need to convert to 16000Hz/np.float32 for RVC input
# VITS output is originally np.float32, so just convert sampling rate
def load_audio(input_audio):
    sr, audio = wavfile.read(input_audio)  # Read audio file
    original_sr = sr
    target_sr = 16000
    original_length = len(audio)
    target_length = int(original_length * (target_sr / original_sr))

    return resample(audio, target_length)


def convert_voice(input_audio, vcid, f0_up_key, f0_method):
    global hubert_model, vc, net_g,global_vcid,tgt_sr,version

    # Check if hubert_model is loaded; if not, load it
    if 'hubert_model' not in globals():
        hubert_model = load_hubert()

    # Check if vc and net_g are loaded; if not, load them
    if 'vc' not in globals() or 'net_g' not in globals():
        vc, net_g = get_vc(vcid)
        global_vcid = vcid

    if global_vcid!=vcid and vcid!='No conversion':
        vc, net_g = get_vc(vcid)
        global_vcid = vcid

    # RVC model speaker id
    sid = 0
    f0_file = None
    index_rate = 1

    try:
        file_index = glob.glob(f"rvc/{vcid}/*.index")[0]
    except:
        file_index = f"rvc/{vcid}/added.index"

    try:
        file_big_npy = glob.glob(f"rvc/{vcid}/*.npy")[0]
    except:
        file_big_npy = f"rvc/{vcid}/total_fea.npy"

    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio)
    times = [0, 0, 0]

    # get from cpt
    if_f0 = cpt.get("f0", 1)
    sr = int(cpt.get("sr", 1).replace("k", "")) * 1000

    # Add the missing arguments with placeholders and comments
    filter_radius = 3  # Placeholder for filter_radius; update with actual value
    resample_sr = tgt_sr    # Placeholder for resampling rate; update with actual value
    rms_mix_rate = 1   # Placeholder for RMS mix rate; update with actual value
    protect = 0.33        # Placeholder for protect; update with actual value

    return sr, vc.pipeline(
        hubert_model, net_g, sid, audio, input_audio, times, f0_up_key, f0_method, 
        file_index, index_rate, if_f0, 
        filter_radius, tgt_sr, resample_sr, rms_mix_rate, version, protect,
        f0_file=f0_file
    )



def load_hubert():
    # global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["rvc/hubert_base.pt"], suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if (is_half):
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

    return hubert_model



def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def text2speech(model_path,config_path, text, vcid, pitch, f0method):
    tts_audio = tts(model_path,config_path, text)
    if vcid != 'No conversion':
        return convert_voice(tts_audio, vcid, pitch, f0method)

    return tts_audio


def tts(model_path, config_path, text):
    model_path = './logs/' + model_path
    config_path = './configs/' + config_path
    hps = utils.get_hparams_from_file(config_path)

    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)

    stn_tst = get_text(text, hps)
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

    with torch.no_grad():
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    output_wav_path = "output.wav"
    write(output_wav_path, hps.data.sampling_rate, audio)

    return output_wav_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file.')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the config file.')
    args = parser.parse_args()

    model_files = [f for f in os.listdir('./logs/') if f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    config_files = [f for f in os.listdir('./configs/') if f.endswith('.json')]

    default_model_file = args.model_path if args.model_path else (model_files[0] if model_files else None)
    default_config_file = args.config_path if args.config_path else 'config.json'

    # The new sliders and radio button
    pitch = gr.Slider(minimum=-12, maximum=12, step=1, label='Pitch', value=0)
    f0method = gr.Radio(label="Pitch Method pm: speed-oriented, harvest: accuracy-oriented", choices=["pm", "harvest"], value="pm")


    # New Dropdown for voice conversion
    vcid = gr.inputs.Dropdown(choices=vc_models, label="Voice Conversion", default='No conversion')

    # Create Interface
    gr.Interface(
        fn=text2speech,
        inputs=[
            components.Dropdown(model_files, value=default_model_file, label="Model File"),
            components.Dropdown(config_files, value=default_config_file, label="Config File"),
            components.Textbox(label="Text Input"),
            vcid,
            gr.Slider(minimum=-20, maximum=20, step=1, label='Pitch', value=0),
            gr.Radio(label="Pitch Method pm: speed-oriented, harvest: accuracy-oriented", choices=["pm", "harvest"], value="pm")
        ],
        outputs=components.Audio(type='filepath', label="Generated Speech"),
        live=False
    ).launch()