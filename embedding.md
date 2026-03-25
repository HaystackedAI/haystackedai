Embeddings are not only semantic.

They also encode:

Syntactic patterns (structure, word order)

Contextual usage

Style, topic, tone (to some extent)

Statistical correlations learned during training

The model doesn’t “understand” meaning in a human sense — it encodes features useful for similarity and downstream tasks.

Embeddings encode statistical correlations in data that approximate semantic, syntactic, and contextual relationships.

How both statements are true
1️⃣ At the learning level (how they’re formed)

Embeddings arise from statistical correlations in data

Learned via objectives like next-token prediction

No explicit semantics

➡️ Correlation is the mechanism

2️⃣ At the representation level (how we use them)

Each dimension encodes some latent property

Those properties function as features for downstream tasks:

similarity search

clustering

classification

retrieval

➡️ Embedding vectors = learned features

Analogy (very accurate)

Pixels → features → concepts

Embeddings are like automatically learned features, not hand-crafted ones

Clean reconciled definition

If you want wording that satisfies both camps:

An embedding is a learned feature representation of text, where each vector dimension encodes latent statistical correlations that approximate semantic and contextual properties.

One-line takeaway

Correlation → how embeddings are learned

Features → how embeddings are used

Both are technically correct, just spoken from different levels of abstraction.


Embedding is a mapping from a discrete / low-information representation to a dense, high-dimensional continuous space.

Examples:

Token ID (single integer)
→ 768 / 1024 / 1536-dim vector

Short text / sparse bag-of-words
→ dense vector

So this statement is correct:

Embedding maps low-dimensional or sparse representations into a higher-dimensional dense space.

Why people say this

Original text is:

discrete

symbolic

sparse

Embedding space is:

continuous

dense

high-dimensional

Higher dimensionality allows:

linear separability

smoother similarity geometry

Why confusion happens

In ML you’ll also hear the opposite:

“Embeddings reduce dimensionality”

That’s true only when compared to extremely high-dimensional sparse vectors
(e.g., one-hot vectors with 100k dimensions).

So both are true depending on the reference point:

From	To	Description
token ID (1D)	768D	dimension expansion
one-hot (100kD)	768D	dimension reduction
Clean, precise definition (recommended)

An embedding is a learned mapping from discrete or sparse inputs into a dense continuous vector space where relationships are preserved.

This wording avoids the low- vs high-dimension trap and is technically solid.






Abstract / Math-focused answer

“An embedding is a mapping from a discrete or structured input space into a continuous vector space, learned such that geometric relationships in the vector space reflect statistical correlations in the original data. Mathematically, embeddings arise as a solution to an optimization problem—often implicit in neural networks—where similarity in the high-dimensional vector space approximates co-occurrence or contextual probabilities. In other words, embeddings are a low-rank factorization of the co-occurrence statistics of the data, preserving relevant relationships while projecting into a dense latent space suitable for downstream computations.”

Key points this shows

Abstract: Talks about mapping from discrete/structured → continuous vector space.

Mathematical origin: Frames embeddings as solutions to an optimization problem.

Relation to data statistics: Connects to co-occurrence / correlation / factorization.

Latent space interpretation: Shows understanding of high-dimensional geometry and feature representation.






