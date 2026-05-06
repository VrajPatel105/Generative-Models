# Self-Supervised Representation Learning — Notes

## What I learned

This week I went deep on representation learning without labels — specifically VAEs, 
GANs, and contrastive methods like SimCLR. The goal was to understand how modern 
vision and embedding models learn useful features from unlabeled data, since this is 
the foundation of almost every production ML system that uses embeddings (RAG, 
retrieval, recommendation, CLIP-style multimodal models).

## The core problem

Labeled data is expensive. The internet has billions of unlabeled images, videos, and 
text documents. Self-supervised learning is the family of techniques that lets us 
turn that unlabeled data into useful representations.

The trick is creating supervision signals from the data itself. Instead of asking 
humans to label "this is a cat", you design a task where the data supervises itself. 
Different methods do this differently.

## Three approaches I studied

**Variational Autoencoders (VAE)** encode an image into a probability distribution 
in latent space, then decode it back. The key trick is the reparameterization trick, 
which lets gradients flow through a sampling operation. VAEs give you a smooth, 
sampleable latent space — useful for understanding generative modeling, and they're 
the conceptual ancestor of diffusion models.

**Generative Adversarial Networks (GAN)** train two networks against each other: a 
generator that tries to produce realistic images, and a discriminator that tries to 
detect fakes. They were dominant for image generation around 2015–2020 but have been 
largely replaced by diffusion models due to training instability and mode collapse 
issues.

**SimCLR (contrastive learning)** is the one I focused on because it's most relevant 
to production ML. The idea: take an image, create two augmented views of it (random 
crop + color jitter), pass both through an encoder, and train the model to produce 
similar embeddings for the two views and dissimilar embeddings for views of other 
images in the batch. No labels needed — the augmentation pipeline creates the 
supervision signal for free.

The architecture has two parts: a backbone encoder (like ResNet-50) and a small 
projection head used only during training. The projection head gets discarded 
afterward; the encoder is what you keep and use for downstream tasks.

The loss function (NT-Xent / InfoNCE) is essentially classification in disguise: for 
each view, you classify which of the other 2N–1 views in the batch is its augmented 
partner, using cosine similarity as the logits. This means batch size matters — more 
views per batch means more negatives, which makes the task harder and the learned 
representations stronger.

## Why this matters in 2026

Almost every embedding model used in production today (BGE, E5, OpenAI's 
text-embedding-3, sentence-transformers, CLIP, DINOv2) is descended from contrastive 
learning. When you build a RAG system, do semantic search, or use a vision foundation 
model, you're using the ideas from SimCLR adapted to the relevant domain.

The biggest practical insight from SimCLR wasn't the architecture — it was the 
empirical finding that the augmentation pipeline is what does the heavy lifting. 
Specifically, the combination of random crop + color distortion is the magic combo 
for images: each augmentation alone leaves shortcuts the model can exploit, but 
together they force the model to learn actual visual content.

## What I'm taking forward

- Foundation models like DINOv2 are pretrained on massive unlabeled datasets using 
  these methods. For domain-specific work, the realistic approach is to start from 
  one of these, do continual self-supervised pretraining on your domain data, then 
  fine-tune on whatever labels you have.
- Contrastive thinking applies far beyond images — it's how sentence embeddings, 
  multimodal models, and many retrieval systems are trained.
- The supervision-from-structure idea is the conceptual core of self-supervised 
  learning, not the specific architectures.