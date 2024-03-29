# Generative-Experiments

## TO DO
* Mode Collapse
    * ~~Test for loss formulation errors?~~
        *  Loss forulation is correct
    * Generator and discriminator information flow?
* Why does everyone use LeakyReLU instead of the normal one?
* Testing with original GAN loss formulation; using BCE; and WGAN loss.
* Discriminator is crushing the generator in the minmax.
* Wasserstein Loss vs. GAN loss? Do I need to add in gradient penalty to balance out
* Convolutional Generator and Discriminator
* WGAN - https://arxiv.org/abs/1701.07875
    * Add gradient penalty
    * Update labels for WGAN training --> (-1,1) instead of (0,1)?
* Traversing z-space images clip
* Conditional GAN for MNIST images with single model; single G and single D for multiple classes

## DONE
* ~~Measure time discriminator updates for isolated random batch. The discriminator runs really slow. Why? Gradients? Optimizer update?~~
    * Discriminator had way too many parameters, approx 10x too many. Reducing feature count dropped the average time of a 128-image-batch from `.312 s` --> `.006 s`.
* ~~Discriminator loss approaches 0, causing vanishing gradients and the generator training to collapse. How to prevent vanishing gradients?~~
    * In this case, I was able to use `BCEWithLogitsLoss` instead of `BCELoss`. Potentially just a numerical stability issue that is implemented in `WithLogits`?