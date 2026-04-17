# Literature Survey on AI-Powered Fashion Recommendation and Outfit Compatibility Using Machine Learning

---

**Course Name and Code:** [Your Course Name and Code]
**Assignment Title:** Literature Survey on AI-Powered Fashion Recommendation and Outfit Compatibility Using Machine Learning
**Group Members:**
- Sameer Saxena — [Student ID]
**Submission Date:** April 17, 2026

---

## Abstract

This report presents a literature survey on machine learning approaches for fashion recommendation and outfit compatibility prediction, motivated by the development of the Smart Wardrobe system — a local, fully offline outfit recommender that extracts visual embeddings from clothing images and scores outfit combinations by both visual compatibility and event suitability. Eight foundational papers are surveyed, spanning visual feature extraction (He et al., 2016), large-scale dataset construction (Liu et al., 2016), sequential compatibility modelling (Han et al., 2017), type-conditioned embedding spaces (Vasileva et al., 2018), personal-closet grading (Tangseng et al., 2017), multi-layer comparison networks (Wang et al., 2019), graph-based context modelling (Cucurull et al., 2019), and visually-aware personalised ranking (He & McAuley, 2016). The survey identifies three core technical challenges in the field: (1) learning transferable visual representations for clothing items, (2) modelling multi-item outfit coherence, and (3) personalising recommendations to user context. The Smart Wardrobe project addresses challenges (1) and (2) using ResNet50 embeddings and cosine similarity scoring, enhanced with a trained event-suitability classifier. Key gaps include the absence of event/occasion labels in existing benchmark datasets and the difficulty of learning compatibility from weakly-paired image data.

---

## 1. Introduction

### Background

Fashion recommendation is a rapidly expanding subdomain of machine learning with direct commercial and personal relevance. With the global fashion e-commerce market exceeding USD 700 billion annually, there is strong demand for intelligent systems that help users identify what items pair well together, suggest complete outfits from a personal wardrobe, and tailor recommendations to specific occasions or style preferences. Historically, recommendation systems in this domain relied on collaborative filtering — inferring preferences from user behaviour logs. However, fashion is fundamentally a visual domain: whether two garments "go together" depends on colour harmony, style coherence, silhouette compatibility, and contextual occasion suitability — none of which are captured by purchase history alone.

The emergence of deep convolutional neural networks (CNNs), particularly the ResNet family (He et al., 2016), has enabled high-quality visual feature extraction from clothing images without domain-specific feature engineering. Combined with large annotated fashion datasets such as DeepFashion (Liu et al., 2016) and the Polyvore Outfits corpus (Han et al., 2017; Vasileva et al., 2018), these tools have driven substantial progress on the outfit compatibility problem over the past decade.

The Smart Wardrobe project is a locally-deployed, privacy-preserving outfit recommendation system. It uses ResNet50 to extract 2048-dimensional embeddings from user-uploaded clothing images, stores them in SQLite, and scores all top–bottom–shoes combinations using both cosine similarity (visual compatibility) and a trained MLP event classifier (occasion suitability). The classifier is trained on the Fashion Product Images dataset (Kaggle/Myntra), which contains 44,000 clothing items labelled with usage categories including Casual, Formal, Sports, Party, and Ethnic.

### Objective

This literature survey aims to:
1. Review foundational and state-of-the-art methods for visual fashion compatibility prediction and outfit recommendation.
2. Identify the datasets, model architectures, and evaluation metrics used in the field.
3. Position the Smart Wardrobe system within the landscape of existing work, identifying where it aligns with and diverges from published approaches.
4. Highlight open research gaps, particularly around event-aware recommendation.

### Scope

Papers were selected from peer-reviewed venues (CVPR, ECCV, ACM MM, AAAI) published between 2016 and 2019. Selection prioritised: (a) direct relevance to visual compatibility or outfit recommendation, (b) availability of reproducible datasets, and (c) citation influence (all selected papers have >100 citations). One foundational backbone paper (ResNet, He et al., 2016) is included as it is the universal visual feature extractor across the entire field.

---

## 2. Methodology

### Search Strategy

Literature was identified using Google Scholar, Semantic Scholar, arXiv, and the ACM Digital Library. Author homepages (e.g., Xintong Han, Mariya Vasileva) and dataset repositories (GitHub: xthan/polyvore-dataset, mvasil/fashion-compatibility) provided additional context and citations.

### Keywords

The following search terms were used:
- "outfit compatibility prediction"
- "fashion recommendation deep learning"
- "visual compatibility embedding"
- "clothing co-purchase neural network"
- "type-aware fashion embedding"
- "wardrobe recommendation neural"
- "ResNet image feature extraction fashion"
- "Polyvore dataset outfit"

### Selection Criteria

| Criterion | Threshold |
|-----------|-----------|
| Venue | Top-tier peer-reviewed (CVPR, ECCV, ACM MM, AAAI, ICCV) |
| Publication year | 2016–2019 (foundational period for deep learning in fashion) |
| Relevance | Direct treatment of visual compatibility or fashion recommendation |
| Reproducibility | Dataset or code publicly available |
| Citation count | >100 Google Scholar citations |

Papers on clothing segmentation, virtual try-on, or garment generation were excluded as they address different sub-problems.

---

## 3. Literature Review

### 3.1 Thematic Analysis

The surveyed literature organises naturally into four themes: (A) Visual Backbone and Feature Extraction, (B) Dataset Construction, (C) Compatibility Modelling Architectures, and (D) Personalised and Contextual Recommendation.

---

#### Theme A: Visual Backbone and Feature Extraction

**He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.**

ResNet introduced skip (residual) connections that allow networks of up to 152 layers to be trained without degradation. Rather than learning a full mapping H(x), each residual block learns F(x) = H(x) − x, with the original input added back via a shortcut. This design solved the vanishing gradient problem that had limited prior networks to ~20 layers and enabled ImageNet top-5 error of 3.57%, surpassing human performance.

ResNet-50 became the universal visual backbone for fashion ML research: every subsequent paper in this survey extracts 2048-dimensional image embeddings using a pre-trained ResNet-50. The Smart Wardrobe system follows this exact convention, using `torchvision.models.resnet50(weights=DEFAULT)` with the classification head removed to produce per-item embeddings.

---

#### Theme B: Dataset Construction

**Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. *CVPR 2016*.**

DeepFashion is the first large-scale benchmark for fashion recognition, containing 800,000+ clothing images across 50 categories, annotated with 1,000 attributes, bounding boxes, and 300,000 cross-domain image pairs (shop vs. consumer). The authors propose FashionNet, a CNN that jointly predicts attributes and spatial landmarks, using estimated landmarks to pool features at garment regions. Top-20 consumer-to-shop retrieval accuracy reached 18.8%.

DeepFashion established the practice of rich multi-attribute annotation in fashion datasets and provided pre-training corpora for subsequent compatibility models. Its attribute vocabulary (colour, texture, style) is directly related to the feature space ResNet embeddings implicitly capture.

**Han, X., Wu, Z., Jiang, Y.-G., & Davis, L. S. (2017). Learning Fashion Compatibility with Bidirectional LSTMs. *ACM MM 2017*.**

This paper introduced the Polyvore dataset: 21,889 outfits (17,316 train, 1,497 val, 3,076 test) scraped from the Polyvore fashion platform, with ground-truth compatible and incompatible outfit labels and fill-in-the-blank questions. The dataset became the standard benchmark for outfit compatibility and is the primary training corpus used in Papers 2, 5, 6, and 7 of this survey. The Smart Wardrobe project uses the Polyvore metadata (compatibility labels and outfit JSONs) as reference for the field.

---

#### Theme C: Compatibility Modelling Architectures

**Han et al. (2017)** — in addition to introducing the Polyvore dataset — proposed the first deep sequential model for outfit compatibility. Treating an outfit as an ordered item sequence (top-to-bottom), a Bidirectional LSTM processes item embeddings in both forward and backward directions to capture directional compatibility dependencies. An auxiliary visual-semantic embedding jointly aligns CNN features with semantic attribute descriptions, adding structured fashion knowledge as regularisation. The model achieved a 19.4% AUC improvement over prior state-of-the-art on compatibility prediction and a 9.7% improvement on fill-in-the-blank.

**Vasileva, M. I., Plummer, B. A., Dusad, K., Rajpal, S., Kumar, R., & Forsyth, D. (2018). Learning Type-Aware Embeddings for Fashion Compatibility. *ECCV 2018*.**

Vasileva et al. observed that a single embedding space conflates two distinct notions: similarity (items that are interchangeable within a category) and compatibility (items from different categories that coordinate aesthetically). They propose a family of type-specific projection matrices, one for each ordered pair of item types (top–bottom, top–shoes, etc.), projecting items into a type-conditioned subspace before computing similarity. Compatibility is thus evaluated only within the appropriate subspace, while a separate loss enforces similarity within types. This achieved AUC ~0.86 and FITB 55.3%, outperforming the sequential Bi-LSTM approach.

**Wang, X., Wu, B., Ye, Y., & Zhong, Y. (2019). Outfit Compatibility Prediction and Diagnosis with Multi-Layered Comparison Network. *ACM MM 2019*.**

Wang et al. decompose outfit compatibility into all pairwise type-specific item comparisons and evaluate them at multiple CNN layers simultaneously — low layers for colour/texture, high layers for style/silhouette — yielding a multi-granularity compatibility score. Uniquely, the framework also supports *compatibility diagnosis*: backpropagating through the comparison network identifies which item pair causes outfit failure. On the Polyvore-T benchmark, it achieved AUC 91.90 and FITB 64.35%, establishing state-of-the-art at the time.

**Cucurull, G., Taslakian, P., & Vazquez, D. (2019). Context-Aware Visual Compatibility Prediction. *CVPR 2019*.**

Cucurull et al. reformulate compatibility as a graph edge prediction problem. Outfit items are graph nodes; a Graph Convolutional Network (GCN) aggregates information from compatible neighbours to produce context-aware embeddings. A white shirt's embedding adapts depending on whether it is being evaluated with formal trousers or casual jeans. With k=15 context neighbours, FITB accuracy reached 96.9% on Polyvore Resampled — a 27.7 percentage-point improvement over prior methods. This represents the highest known accuracy on this benchmark.

---

#### Theme D: Personalised and Contextual Recommendation

**He, R., & McAuley, J. (2016). VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. *AAAI 2016*.**

VBPR extends Bayesian Personalised Ranking (BPR) by incorporating visual product features. CNN features are projected into a low-dimensional visual preference space; each user learns both a latent collaborative taste vector and a visual taste vector. The final ranking score combines collaborative filtering signals with a visual compatibility dot product. This enables cold-start recommendation for unseen items and achieved AUC ~0.77 on Amazon fashion data — a significant improvement over non-visual baselines.

**Tangseng, P., Yamaguchi, K., & Okatani, T. (2017). Recommending Outfits from Personal Closet. *ICCV 2017 Workshops*.**

Tangseng et al. directly address the real-world wardrobing scenario: given a personal collection of clothing items, score and recommend the best outfit combinations. Their neural grader takes a variable-length bag of item embeddings, handles missing items via mean embedding, and outputs a compatibility score. Trained on Polyvore409k (409,776 outfits), the model achieved 84% classification accuracy and 91% agreement with human judgements — the highest human-matching rate of any surveyed paper.

---

### 3.2 Comparative Analysis

| Paper | Model Type | Dataset | Compat. AUC | FITB Acc. | Event Labels? | Personalised? |
|-------|-----------|---------|-------------|-----------|---------------|---------------|
| Han et al. 2017 | Bi-LSTM | Polyvore 21K | +19.4% vs. SOTA | +9.7% vs. SOTA | No | No |
| Vasileva et al. 2018 | Type-conditioned MLP | Polyvore 68K | ~0.86 | 55.3% | No | No |
| Wang et al. 2019 | Multi-layer CNN comparison | Polyvore-T | 91.90 | 64.35% | No | No |
| Cucurull et al. 2019 | Graph Conv. Network | Polyvore / Amazon | — | 96.9% | No | No |
| Tangseng et al. 2017 | Flat MLP | Polyvore 409K | — | — | No | Partial (closet) |
| He & McAuley 2016 | BPR + Visual MF | Amazon Fashion | AUC ~0.77 | — | No | Yes |
| **Smart Wardrobe** | ResNet + Cosine + MLP | Fashion Product Images 44K | Baseline cosine | — | **Yes** | No |

A key observation from the comparison table is that **no existing benchmark dataset provides event/occasion labels**. All Polyvore-based papers evaluate pure aesthetic compatibility (does this outfit look good together?) without considering occasion suitability (is this outfit appropriate for a wedding vs. the gym?). This is the primary differentiating contribution of the Smart Wardrobe project: by training on the Fashion Product Images dataset, which includes `usage` labels (Casual, Formal, Sports, Party, Ethnic), the system explicitly models event suitability as a separate scoring dimension alongside visual compatibility.

A second key difference is dataset scale and nature. Polyvore provides *outfit-level* compatibility labels (this set of items is a good outfit), while Fashion Product Images provides *item-level* occasion labels (this item is Casual/Formal/etc.). The Smart Wardrobe system leverages item-level labels as the training signal for an event classifier, then combines this with pairwise cosine similarity for the final outfit score.

---

## 4. Critical Analysis

### 4.1 Evaluation of Research Quality

The Polyvore dataset papers (Han et al. 2017, Vasileva et al. 2018, Wang et al. 2019, Cucurull et al. 2019) share a common, well-established evaluation protocol — compatibility AUC and fill-in-the-blank accuracy — enabling direct comparison. However, the dataset was scraped from a fashion-focused social platform, which introduces selection bias: Polyvore outfits represent the aesthetic preferences of a specific community of fashion enthusiasts and may not generalise to everyday wardrobe choices.

VBPR (He & McAuley 2016) uses implicit feedback (Amazon purchase logs), which is a more realistic proxy for real-world preferences but conflates style with practicality (a user might buy running shoes without pairing them with anything). Tangseng et al. achieve impressive human-matching rates but rely on the same Polyvore data, making their results contingent on the same biases.

The ResNet paper (He et al. 2016) has exceptional validity: it achieved reproducible state-of-the-art on the well-controlled ImageNet benchmark, and the generalisation of learned features to downstream fashion tasks has been extensively validated in subsequent work.

### 4.2 Identification of Gaps

**Gap 1: No event/occasion labels in existing benchmarks.** All Polyvore-based datasets lack occasion metadata. Fashion compatibility is not context-free — a compatible outfit for a beach holiday may be entirely inappropriate for a job interview. The Fashion Product Images dataset partially fills this gap with usage labels but lacks outfit-level pairing information.

**Gap 2: Limited generalisation across cultures.** Most Polyvore outfits reflect Western fashion aesthetics. The Ethnic category in Fashion Product Images suggests that cultural fashion norms (e.g., Indian occasion wear) are underrepresented in mainstream benchmarks.

**Gap 3: Cold-start for new wardrobe items.** VBPR addresses cold-start at inference via visual embeddings, but most compatibility models (Bi-LSTM, type-aware) require item-type metadata that may not be available for user-uploaded photos.

**Gap 4: Personalisation.** With the exception of VBPR and partially Tangseng et al., the surveyed models do not personalise recommendations to individual users. Wardrobe recommendation is inherently personal — the same outfit may be perfect for one user and wrong for another.

**Gap 5: Dynamic outfit length.** Most models assume fixed item types (top, bottom, shoes). Real wardrobe outfits include accessories, outerwear, and varying item counts; few papers handle variable-length outfit composition gracefully.

### 4.3 Implications

**Practical:** The success of cosine similarity over ResNet embeddings (Tangseng et al., baseline; Smart Wardrobe system) suggests that a strong visual backbone alone provides meaningful compatibility signal without supervised outfit labels — important for deployment in settings where labelled data is unavailable. The 71.8% event classification accuracy achieved in the Smart Wardrobe project on Fashion Product Images demonstrates that item-level occasion labels are a viable training signal for event-aware recommendation, even without outfit-level pairing data.

**Theoretical:** The progression from global cosine similarity → type-aware subspaces → multi-layer comparison → graph-based context modelling reflects the field's growing understanding that compatibility is a relational, multi-granularity, and context-dependent property. This has implications beyond fashion, applying to any domain where visual coherence between multiple objects must be assessed.

### 4.4 Limitations of This Survey

- The survey is limited to English-language publications from Western academic venues; significant work in Chinese-language venues (especially ACM MM) may be underrepresented.
- Papers after 2019 (e.g., transformer-based approaches, CLIP-powered fashion recommendation) are outside the selected scope.
- The Smart Wardrobe system's event classifier is evaluated on a held-out split of Fashion Product Images; no cross-dataset or out-of-distribution evaluation has been performed.
- Citation counts reflect academic impact rather than practical deployment success.

---

## 5. Conclusion

### Summary of Findings

This survey reviewed eight foundational papers spanning visual feature extraction, dataset construction, compatibility modelling, and personalised recommendation in fashion ML. Key findings are:

1. **ResNet50 embeddings** (He et al., 2016) are the universal visual backbone and provide meaningful compatibility signals even without domain-specific training.
2. **Sequential models** (Han et al., 2017) were the first deep approach but have since been outperformed by type-aware (Vasileva et al., 2018), multi-layer comparative (Wang et al., 2019), and graph-based context models (Cucurull et al., 2019).
3. **Pairwise cosine similarity** remains a strong and practical baseline when labelled compatibility data is unavailable.
4. **No existing dataset provides event/occasion labels** at the outfit level, leaving event-aware recommendation as an open research problem.
5. The **Fashion Product Images dataset** fills this gap at the item level, enabling training of an event suitability classifier — the approach adopted by the Smart Wardrobe system.
6. **Personalisation** (VBPR, Tangseng et al.) requires either purchase history or a personal closet, and represents the next natural extension of the Smart Wardrobe system.

### Future Directions

1. **Event-labelled outfit datasets.** Creating a benchmark dataset of outfits annotated with occasion suitability labels would enable end-to-end training of event-aware compatibility models, replacing the current item-level approximation.
2. **Transformer-based outfit encoding.** Vision Transformers (ViTs) and CLIP (Radford et al., 2021) have demonstrated superior zero-shot visual-semantic alignment compared to CNNs and could replace ResNet as the visual backbone for richer compatibility representations.
3. **User personalisation.** Incorporating user feedback (liked/disliked outfits) into the Smart Wardrobe recommender via a VBPR-style personalisation layer would enable style adaptation over time.
4. **Explainability.** Adopting the diagnostic framework of Wang et al. (2019) would allow the system to explain *why* an outfit is flagged as incompatible — e.g., "the shoes clash with the top's colour palette" — making recommendations actionable.
5. **Cultural diversity.** Extending training data to include culturally diverse fashion datasets would improve recommendation quality for non-Western clothing styles.

---

## 6. References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, pp. 770–778. IEEE/CVF. https://arxiv.org/abs/1512.03385

[2] He, R., & McAuley, J. (2016). VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. *Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016)*, vol. 30, no. 1, pp. 144–150. AAAI Press. https://arxiv.org/abs/1510.01784

[3] Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, pp. 1096–1104. IEEE/CVF.

[4] Han, X., Wu, Z., Jiang, Y.-G., & Davis, L. S. (2017). Learning Fashion Compatibility with Bidirectional LSTMs. *Proceedings of the 25th ACM International Conference on Multimedia (ACM MM 2017)*, pp. 1078–1086. ACM. https://arxiv.org/abs/1707.05691

[5] Tangseng, P., Yamaguchi, K., & Okatani, T. (2017). Recommending Outfits from Personal Closet. *Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCV 2017 Workshops)*, pp. 2079–2088. IEEE. https://arxiv.org/abs/1804.09979

[6] Vasileva, M. I., Plummer, B. A., Dusad, K., Rajpal, S., Kumar, R., & Forsyth, D. (2018). Learning Type-Aware Embeddings for Fashion Compatibility. *Proceedings of the European Conference on Computer Vision (ECCV 2018)*, LNCS vol. 11219, pp. 390–405. Springer. https://arxiv.org/abs/1803.09196

[7] Wang, X., Wu, B., Ye, Y., & Zhong, Y. (2019). Outfit Compatibility Prediction and Diagnosis with Multi-Layered Comparison Network. *Proceedings of the 27th ACM International Conference on Multimedia (ACM MM 2019)*. ACM. https://arxiv.org/abs/1907.11496

[8] Cucurull, G., Taslakian, P., & Vazquez, D. (2019). Context-Aware Visual Compatibility Prediction. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019)*, pp. 12617–12626. IEEE/CVF. https://arxiv.org/abs/1902.03646

[9] Kaggle — Fashion Product Images Dataset (Myntra). https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

---

## 7. Appendix

### A. Smart Wardrobe System Architecture

```
User uploads clothing image (top / bottom / shoes)
              ↓
   ResNet50 extracts 2048-dim embedding
   (torchvision pretrained on ImageNet)
              ↓
   Embedding + thumbnail stored in SQLite
              ↓
   User selects event (casual / office / wedding / party / date night / gym)
              ↓
   All wardrobe combinations enumerated (tops × bottoms × shoes)
              ↓
   ┌──────────────────────────────────┐
   │  Compatibility Score             │
   │  Cosine similarity (pairwise)    │  ← Baseline from Han et al. / Tangseng et al.
   └──────────────────────────────────┘
               +
   ┌──────────────────────────────────┐
   │  Event Suitability Score         │
   │  MLP classifier (2048→512→128→7) │  ← Trained on Fashion Product Images
   │  Trained on usage labels         │    (Casual, Formal, Sports, Party, etc.)
   └──────────────────────────────────┘
              ↓
   Final Score = 0.5 × Compat + 0.5 × Event
              ↓
   Top 3 outfits returned and displayed in Streamlit UI
```

### B. Event Classifier Training Results

| Metric | Value |
|--------|-------|
| Training samples | 35,280 |
| Validation samples | 8,821 |
| Classes | 7 (Home excluded — 1 sample) |
| Best validation accuracy | **71.8%** |
| Epochs trained | 30 (no early stopping triggered) |
| Architecture | MLP: 2048 → 512 (BN+ReLU+Drop) → 128 (BN+ReLU+Drop) → 7 |
| Optimiser | Adam, lr=1e-3, ReduceLROnPlateau |
| Class weighting | Inverse frequency (handles Casual dominance) |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Casual | 0.99 | 0.65 | 0.79 |
| Ethnic | 0.58 | 0.98 | 0.73 |
| Formal | 0.47 | 0.95 | 0.63 |
| Party | 0.19 | 0.50 | 0.27 |
| Smart Casual | 0.05 | 0.23 | 0.08 |
| Sports | 0.35 | 0.95 | 0.52 |
| Travel | 0.29 | 0.80 | 0.42 |

The high recall with lower precision for minority classes reflects the class-weighted training strategy, which prioritises detecting underrepresented events at the cost of some false positives.

### C. Dataset Summary

| Dataset | Items / Outfits | Labels | Used In |
|---------|----------------|--------|---------|
| Fashion Product Images | 44,101 items | usage (7 classes) | Event classifier training |
| Custom wardrobe dataset | 600 items (200 tops, 200 bottoms, 200 shoes) | category | Inference / recommendation |
| Polyvore (reference) | 21,889 outfits | compatibility (binary) | Literature context only |
