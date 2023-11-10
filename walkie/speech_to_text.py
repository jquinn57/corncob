import torch
import argparse

class SpeechToText():
    def __init__(self):

        self.device = torch.device('cuda')  # gpu also works, but our models are fast enough for CPU
        self.model, self.decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                        model='silero_stt',
                                        language='en', # also available 'de', 'es'
                                        device=self.device)
        (self.read_batch, split_into_batches,
        read_audio, self.prepare_model_input) = utils  # see function signature for details

    def process_wav(self, filename):
        input = self.prepare_model_input(self.read_batch([filename]), device=self.device)
        output =self.model(input)
        out_text = self.decoder(output[0].cpu())
        return out_text



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="NOAA APT Processor")
    parser.add_argument('--input', default=None, help='Path to wav file')
    args = parser.parse_args()

    stt = SpeechToText()
    print(stt.process_wav(args.input))