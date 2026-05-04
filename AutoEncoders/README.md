# AudoEncoders


## what they are
 
basically an architecture with two parts: encoder and decoder. the encoder compresses the input down into a small latent space (compressed representation), and the decoder takes that latent vector and tries to reconstruct the original input from it.
 
 ![alt text](images/image.png)

## the main problem
 
autoencoders dont really work for generation. heres why:
 
- if you take a latent vector that actually came from the encoder, the decoder reconstructs it well. makes sense, thats what it was trained on.
- but if you give it a random vector, the reconstruction is soo bad.
- even a vector thats *close* to one of the encoded points still doesnt reconstruct properly.
- so the latent space only "works" at the exact points the encoder mapped during training. everywhere else is broken.
## why this actually happens (refinement)
 
the real reason is that the latent space is not regularized. during training, the AE only optimizes reconstruction loss, so the encoder has zero incentive to organize the latent space in any meaningful way.
 
what this means in practice:
 
- two similar inputs can get mapped to wildly different latent vectors
- two nearby latent vectors can decode to completely unrelated outputs
- the space is full of "holes", regions the decoder has never seen
- theres no smooth continuous structure
so the problem isnt just about sampling outside the training distribution. its that the latent space itself has no usable structure for generation. discontinuous, scattered, full of gaps.
 
## what autoencoders are still good for
 
the encoder part is genuinely useful, just not for generation:
 
- **compression** (the main use case)
- **denoising** (denoising autoencoders)
- **anomaly detection** (high reconstruction error = probably out of distribution)
- **feature learning** for downstream tasks

Here, we code audo encoders from scratch
our results: 
![!\[alt text\](reconstruction_results.png)](images/reconstruction_results.png)


## next up: VAE
 
quick teaser for what fixes this. VAEs:
 
- make the encoder output a *distribution* (mean and variance) instead of a single point
- add a KL divergence term that pulls all those distributions toward a standard gaussian
- result: latent space becomes approximately a unit gaussian. dense, smooth, easy to sample from.
thats the whole trick. regularization of the latent space.


