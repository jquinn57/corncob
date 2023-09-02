import torch
#import torchaudio

class SpeechToText():
    def __init__(self):

        self.device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
        self.model, self.decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                        model='silero_stt',
                                        jit_model='jit_xlarge',
                                        language='en', # also available 'de', 'es'
                                        device=self.device)
        (self.read_batch, split_into_batches,
        read_audio, self.prepare_model_input) = utils  # see function signature for details

    def process_wav(self, filename):
        input = self.prepare_model_input(self.read_batch([filename]), device=self.device)
        output =self.model(input)
        out_text = self.decoder(output[0].cpu())
        return out_text