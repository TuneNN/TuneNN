import os

import audioflux as af
import numpy as np


class FeatureExtract:

    def __init__(self, cfg):
        stft_cfg = cfg['feature']['stft']
        bark_cfg = cfg['feature']['bark']
        cep_cfg = cfg['feature']['cep']
        cqt_cfg = cfg['feature']['cqt']

        self.cfg = cfg
        self.sample_rate = sr = cfg['feature']['sample_rate']
        self.stft_obj = af.BFT(samplate=sr, scale_type=af.type.SpectralFilterBankScaleType.LINEAR, **stft_cfg)
        self.bark_obj = af.BFT(samplate=sr, scale_type=af.type.SpectralFilterBankScaleType.BARK, **bark_cfg)
        self.cep_obj = af.Cepstrogram(samplate=sr, **cep_cfg)
        self.cqt_obj = af.CQT(samplate=sr, **cqt_cfg)

    def extract_stft(self, audio_arr):
        spec_arr = self.stft_obj.bft(audio_arr)
        spec_arr = np.abs(spec_arr)
        spec_arr = af.utils.power_to_db(spec_arr)
        return spec_arr

    def extract_bark(self, audio_arr):
        spec_arr = self.bark_obj.bft(audio_arr)
        spec_arr = np.abs(spec_arr)
        spec_arr = af.utils.power_to_db(spec_arr)
        return spec_arr

    def extract_cep(self, audio_arr):
        cep_arr, _, _ = self.cep_obj.cepstrogram(audio_arr)
        cep_arr = cep_arr[1:, ]
        return cep_arr

    def except_cqhc(self, audio_arr):
        hc_num = self.cfg['feature']['cqhc']['hc_num']
        spec_arr = self.cqt_obj.cqt(audio_arr)
        cqhc_arr = self.cqt_obj.cqhc(spec_arr, hc_num)
        return cqhc_arr

    def load_audio(self, audio_path):
        if audio_path is None:
            audio_path = self.cfg['audio_path']

        ret = []
        for name in os.listdir(audio_path):
            if not name.endswith('.wav'):
                continue
            audio_fp = os.path.join(audio_path, name)
            ret.append(audio_fp)
        return ret

    def run(self, audio_path=None, save_path=None):
        audio_list = self.load_audio(audio_path)
        if save_path is None:
            save_path = self.cfg['save_path']

        for audio_fp in audio_list:
            audio_arr, sr = af.read(audio_fp, samplate=self.sample_rate)

            stft_arr = self.extract_stft(audio_arr)
            bark_arr = self.extract_bark(audio_arr)
            cep_arr = self.extract_cep(audio_arr)
            cqhc_arr = self.except_cqhc(audio_arr)

            name = os.path.basename(audio_fp)
            name = os.path.splitext(name)[0]
            file_path = os.path.join(save_path, name)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            stft_fp = os.path.join(file_path, f'{name}_stft.npy')
            bark_fp = os.path.join(file_path, f'{name}_bark.npy')
            cep_fp = os.path.join(file_path, f'{name}_cep.npy')
            cqhc_fp = os.path.join(file_path, f'{name}_cqhc.npy')

            print(stft_arr.shape)
            print(bark_arr.shape)
            print(cep_arr.shape)
            print(cqhc_arr.shape)

            np.save(stft_fp, stft_arr)
            np.save(bark_fp, bark_arr)
            np.save(cep_fp, cep_arr)
            np.save(cqhc_fp, cqhc_arr)


if __name__ == '__main__':
    setting = {
        'feature': {
            'sample_rate': 32000,
            'stft': {'num': 2048},
            'bark': {'num': 128},
            'cep': {},
            'cqt': {'num': 84},
            'cqhc': {'hc_num': 16},
        }
    }
    obj = FeatureExtract(cfg=setting)
    obj.run('./test_wav', './feature')
