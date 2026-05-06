# Self-Supervised & Generative Representation Learning

phase 2, week 3 of my ML roadmap. covers learning useful representations from unlabeled data — autoencoders, VAEs, and contrastive methods (SimCLR).

## why this matters

most ML these days runs on top of pretrained representations. RAG embeddings, CLIP, DINOv2, sentence transformers, recommendation systems, basically every embedding model in production was trained using some version of the ideas in this repo. supervised learning gets you classifiers. self-supervised learning gets you the foundation models everything else is built on.

labels are expensive. the internet has billions of unlabeled images and trillions of words. self-supervised learning is how you turn that into something useful.

## whats inside

### 1. AutoEncoders/

basic autoencoder built from scratch in pytorch on MNIST. encoder compresses input into a small latent space, decoder reconstructs it. trained with MSE loss.

main takeaway: regular autoencoders are great for compression and feature learning, but completely broken for generation. the latent space has no structure. if you sample a random point you get garbage out. there are holes everywhere, similar inputs map to wildly different latent vectors, and the encoder has zero incentive to organize the space.

this section sets up the problem that VAE solves.

### 2. Variational AutoEncoders/

VAE from scratch. fixes the autoencoder generation problem by making the encoder output a distribution (μ, σ) instead of a single point, and adding a KL divergence term to the loss that pulls every encoded distribution toward N(0, I).

key things implemented:
- probabilistic encoder (two heads for mu and logvar)
- reparameterization trick: `z = mu + sigma * epsilon` so gradients flow through sampling
- combined loss: BCE reconstruction + KL divergence

result: latent space becomes smooth and continuous. random sampling actually works. you can sample z ~ N(0, I) and the decoder produces something that looks like a digit. tradeoff is the outputs are blurry compared to GANs/diffusion because the KL term forces overlapping distributions which averages out sharp details.

### 3. SimCLR/

notes and writeup on contrastive learning. didnt implement from scratch since it wasnt critical for my path forward, but went deep on the concepts.

main idea: take an image, generate two augmented views (random crop + color jitter), train an encoder to produce similar embeddings for the two views and dissimilar embeddings for views of other images in the batch. no labels needed. the augmentation pipeline IS the supervision signal.

key insights:
- the supervision comes from the pairing structure (which views came from the same source) not from any labels
- architecture has two parts, an encoder (kept) and a projection head (discarded after training)
- the InfoNCE loss is just classification in disguise: out of 2N-1 candidates, predict which one is your augmented partner
- batch size matters because more negatives = harder task = better representations
- the augmentation choice is what does most of the work, not the architecture. random crop alone fails. color jitter alone fails. together they work because they break shortcuts and force the model to learn actual content

contains:
- `README.md` — notes on what i learned, three approaches compared
- `writeup.md` — 500-word explanation of why contrastive learning works
