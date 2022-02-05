import torch
import torch.nn as nn
import torch.nn.functional as F



class VectorQuantizerEMA(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay = 0.99, eps = 1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

    def forward(self, inputs):
        # Pytorch video format B,C,H,W
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0,2,3,1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distance
        distances = (torch.sum(flat_input**2, dim = 1, keepdim = True)
                     + torch.sum(self.embedding.weight**2, dim = 1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(out_channel, in_channel, 1),
        )

    def forward(self, x):
        return self.conv(x) + x


class Encoder(nn.Module):
    def __init__(self, in_channel, channel,\
                            n_res_block, n_res_channel,\
                             stride):
        super(Encoder, self).__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU(True),
                nn.Conv2d(channel // 2, channel, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU(True),
                nn.Conv2d(channel, channel, kernel_size = 3, padding = 1),
            ]
        
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU(True),
                nn.Conv2d(channel // 2, channel, kernel_size = 3, padding = 1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(in_channel = channel, out_channel = n_res_channel))
        
        blocks.append(nn.ReLU(True))

        self.blocks = nn.Sequential(*blocks)


    def forward(self, x):
        return self.blocks(x)



class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel,\
        n_res_block, n_res_channel, stride,
    ):
        super(Decoder, self).__init__()

        blocks = [nn.Conv2d(in_channel, channel, kernel_size = 3, padding = 1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(in_channel = channel, out_channel = n_res_channel))

        blocks.append(nn.ReLU(True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, kernel_size = 4, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(channel // 2, out_channel, kernel_size = 4, stride=2 , padding = 1),
                ]
            )
        
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, kernel_size = 4, stride = 2, padding = 1)
                )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class VQVAE(nn.Module):
    def __init__(
        self, in_channel = 3, channel = 128, n_res_block = 2, n_res_channel = 32,\
                embedding_dim = 64, num_embeddings = 512, commitment_cost = 0.25, decay = 0.99
    ):
        super(VQVAE, self).__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride = 4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride = 2)
        self.pre_vq_conv_t = nn.Conv2d(channel, embedding_dim, kernel_size=1, stride=1)
        self.vq_vae_t = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.dec_t = Decoder(embedding_dim, embedding_dim, channel, n_res_block, n_res_channel, stride = 2)

        self.pre_vq_conv_b = nn.Conv2d(embedding_dim + channel, embedding_dim, kernel_size=1, stride=1)
        self.vq_vae_b = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)

        self.upsample_t = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size = 4, stride = 2, padding = 1)
        self.dec = Decoder(
            embedding_dim + embedding_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride = 4,
        )

    def forward(self, x):
        quant_t, quant_b, loss, _, _ = self.encode(x)
        x_recon = self.decode(quant_t, quant_b)

        return x_recon, loss

    
    def encode(self, x):
        enc_b = self.enc_b(x)
        enc_t = self.enc_t(enc_b)

        quant_t = self.pre_vq_conv_t(enc_t)
        loss_t, quant_t, perplexity_t, encodings_t = self.vq_vae_t(quant_t)
        loss_t = loss_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.pre_vq_conv_b(enc_b)
        loss_b, quant_b, perplexity_b, encodings_b = self.vq_vae_b(quant_b)
        loss_b = loss_b.unsqueeze(0)

        return quant_t, quant_b, loss_t + loss_b, encodings_b, encodings_t

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.vq_vae_t()
        return 0 