import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gst import GST
from .modules import PostNet, VarianceAdaptor, stylePredictor, \
    emotionalPredictor, intensityPredictor
from utils.tools import get_mask_from_lengths
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from .heteroGraph import HGT
from sentence_transformers import SentenceTransformer

class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "lstransformer":
        #     from .transformers.lstransformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "fastformer":
        #     from .transformers.fastformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "conformer":
        #     from .transformers.conformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "reformer":
        #     from .transformers.reformer import TextEncoder, Decoder
        else:
            raise ValueError("Unsupported Block Type: {}".format(model_config["block_type"]))

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = self.emotion_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )

            # 情感强度部分
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "score.json"
                ),
                "r",
            ) as f:
                n_intensity = len(json.load(f))
            self.intensity_emb = nn.Embedding(
                n_intensity,
                model_config["transformer"]["encoder_hidden"],  # 256
            )

        self.history_type = model_config["history_encoder"]["type"]

        self.text_embbeder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        data = HeteroData()
        data["audio"], data["emotion"], data["speaker"], data["text"], data["intensity"]

        data["audio", "to", "emotion"], data["audio", "to", "speaker"], data["audio", "to", "text"],
        data["emotion", "to", "speaker"], data["emotion", "to", "text"], data["speaker", "to", "text"],
        data["audio", "to", "audio"], data["emotion", "to", "emotion"], data["speaker", "to", "speaker"],
        data["text", "to", "text"],data["audio", "to", "intensity"],data["intensity", "to", "speaker"],
        data["intensity", "to", "text"],data["intensity", "to", "intensity"],

        data["emotion", "rev_to", "audio"], data["speaker", "rev_to", "audio"], data["text", "rev_to", "audio"],
        data["speaker", "rev_to", "emotion"], data["text", "rev_to", "emotion"], data["text", "rev_to", "speaker"],
        data["intensity", "rev_to", "audio"],data["speaker", "rev_to", "intensity"], data["text", "rev_to", "intensity"]

        # data = T.ToUndirected()(data)
        self.hgt = HGT(hidden_channels=384, out_channels=384, num_heads=2, num_layers=1, data=data)

        self.styleExtractor = GST(256,7)
        self.stylePredictor = stylePredictor(query_dim=512, key_dim=384, num_units=256, num_heads=2)

        self.emotionalPredictor = emotionalPredictor(d_model=384, kernel_size=3, hidden_size=256)
        self.intensityPredictor = intensityPredictor(d_model=384, kernel_size=3, hidden_size=256)


    def forward(
        self,
        basenames,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        emotions=None,
        intensities=None,
        history_info=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):


        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        texts, text_embeds = self.encoder(texts, src_masks)

        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]

        emotion_embeds = None
        if self.emotion_emb is not None:
            emotion_embeds = self.emotion_emb(emotions)


        style = current_style = None

        emotion = None

        intensity = None

        for index in range(len(history_info[1].tolist())):
            cureent_len = history_info[1][index].item()
            if (cureent_len != 0):

                history_text_emb = history_info[2][index][:cureent_len, :]
                history_speaker = history_info[3][index][:cureent_len]
                history_speaker_emb = self.speaker_emb(history_speaker.unsqueeze(0)).squeeze(0)
                history_emotion = history_info[4][index][:cureent_len]
                history_emotion_emb = self.emotion_emb(history_emotion.unsqueeze(0)).squeeze(0)
                history_intensity = history_info[5][index][:cureent_len]
                history_intensity_emb = self.intensity_emb(history_intensity.unsqueeze(0).squeeze(0))

                # print(history_info[6][i])
                history_wav_feature_emb = None
                for history_wav_feature in history_info[6][index]:  # 6:mel

                    history_wav_feature = torch.tensor(history_wav_feature).to(texts[index].device)
                    history_wav_feature = self.styleExtractor(history_wav_feature.unsqueeze(0))[0]

                    if (history_wav_feature_emb == None):
                        history_wav_feature_emb = history_wav_feature
                    else:
                        history_wav_feature_emb = torch.cat([history_wav_feature_emb, history_wav_feature],0)


                current_text_emb = history_info[0][index].unsqueeze(0)

                data = HeteroData()

                data["audio"].x = history_wav_feature_emb
                data["emotion"].x = history_emotion_emb
                data["intensity"].x = history_intensity_emb
                data["speaker"].x = torch.cat([history_speaker_emb, speaker_embeds[index].unsqueeze(0)], 0)

                data["text"].x = torch.cat([history_text_emb, current_text_emb], 0)


                edge = []
                for i in range(data["emotion"].x.shape[0]):
                    for j in range(data["audio"].x.shape[0]):
                        edge.append([j, i])
                data["audio", "to", "emotion"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["audio", "to", "intensity"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["audio", "to", "audio"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["emotion", "to", "emotion"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)

                edge = []
                for i in range(data["speaker"].x.shape[0]):
                    for j in range(data["audio"].x.shape[0]):
                        edge.append([j, i])
                data["audio", "to", "speaker"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["audio", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["emotion", "to", "speaker"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["emotion", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["intensity", "to", "speaker"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["intensity", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)

                edge = []
                for i in range(data["text"].x.shape[0]):
                    for j in range(data["speaker"].x.shape[0]):
                        edge.append([j, i])
                data["speaker", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data["speaker", "to", "speaker"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
                data['text', 'to', 'text'].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)



                data = T.ToUndirected()(data)
                data, self.hgt = data.to(texts[index].device), self.hgt.to(texts[index].device)
                out_text, out_emotion, out_intensity = self.hgt(data.x_dict, data.edge_index_dict)

                out_text = out_text[:-1]

                text_emb = current_text_emb.unsqueeze(0)
                q = text_emb.to(texts[index].device)
                k = v = out_text.unsqueeze(0).to(texts[index].device)

                if (style == None):

                    style = self.stylePredictor(q, k, v)[0]  # [1,1,256]
                    emotion = self.emotionalPredictor(out_emotion.unsqueeze(0).unsqueeze(-1))
                    intensity = self.intensityPredictor(out_intensity.unsqueeze(0).unsqueeze(-1))


                    if (mels != None):
                        current_style = self.styleExtractor(
                            mels[index][:mel_lens[index].item(), :].unsqueeze(0))[0].unsqueeze(0)

                else:

                    style = torch.cat([style, self.stylePredictor(q, k, v)[0]], 0)
                    emotion = torch.cat([emotion, self.emotionalPredictor(out_emotion.unsqueeze(0).unsqueeze(-1))], 0)
                    intensity = torch.cat(
                        [intensity, self.intensityPredictor(out_intensity.unsqueeze(0).unsqueeze(-1))], 0)


                    if (mels != None):
                        current_style = torch.cat([current_style, self.styleExtractor(
                            mels[index][:mel_lens[index].item(), :].unsqueeze(0))[0].unsqueeze(0)], 0)


            else:
                if (style == None):
                    style = torch.zeros(1, 1, 256).to(texts[index].device)
                    intensity = torch.zeros(1, 3).to(texts[index].device)
                    emotion = torch.zeros(1, 7).to(texts[index].device)
                else:
                    style = torch.cat([style, torch.zeros(1, 1, 256).to(texts[index].device)], 0)
                    intensity = torch.cat([intensity, torch.zeros(1, 3).to(texts[index].device)], 0)
                    emotion = torch.cat([emotion, torch.zeros(1, 7).to(texts[index].device)], 0)

                if (current_style == None):
                    current_style = torch.zeros(1, 1, 256).to(texts[index].device)
                else:
                    current_style = torch.cat([current_style, torch.zeros(1, 1, 256).to(texts[index].device)], 0)


        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
            intensity_emdedding,
            emotion_emdedding
        ) = self.variance_adaptor(
            intensity,
            emotion,
            style,
            speaker_embeds,
            emotion_embeds,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output


        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            style,
            current_style,
            intensities,
            intensity_emdedding,
            emotions,
            emotion_emdedding,
            p_targets,
            e_targets,
        )
