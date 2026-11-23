# **Machine Learning for Random Number Generator Predictability and Entropy Analysis**

## **1\. Project overview and engineering problem**

Random number generators (RNGs) are used in cryptography, authentication, secure communications, simulation, and quantum-information systems. RNG implementations can introduce bias or statistical dependence through device noise, readout effects, or implementation errors. Assessing these properties is a practical engineering problem in hardware security, measurement processing, and computational reliability.

Engineering question: *Can a machine-learning pipeline identify statistical fingerprints in RNG outputs and detect next-bit predictability, and how does any detected predictability compare with conventional entropy measurements?*

This project has two separate supervised learning tasks, both built from the **same publicly obtained binary-sequence dataset**:

1. **Task A — Generator-source classification (initial homework dataset):** Extract interpretable numerical features from each 100-bit sequence. Predict the original **`source_label` (1, 2, or 3\)**. Apply PCA and compare classifiers using accuracy, macro-precision, macro-recall, and macro-F1.  
2. **Task B — Next-bit predictability (main follow-up research experiment):** Use the preceding 16 bits to predict the **actual next bit, `next_bit` (0 or 1\)**. Compare models to a source-specific bit-frequency baseline and relate held-out predictive cross-entropy to empirical marginal Shannon entropy.

**Crucial distinction:** Successfully identifying a sequence's *source* is **not** equivalent to predicting its *next bit* or establishing low entropy. The two targets must not be confused.

## **2\. Original dataset: what is it?**

| Property | Description |
| :---- | :---- |
| Original file | `AI_2qubits_training_data.txt` |
| Type | Binary output sequences attributed to quantum-RNG research, including simulated or noise-model-based experiments |
| Observations | **6,000** sequences |
| Sequence length | **100 binary digits** per observation |
| Total digits | **600,000** bits |
| Original classes | **3 generator-source labels** (`1`, `2`, `3`) |
| Class counts | **2,000 sequences per label** |
| Original layout | One **100-character bitstring** followed by a space and **one integer source label** per line |
| Missing/invalid records | None detected in the provided copy; all 6,000 rows have exactly 100 binary digits and one valid label |
| Collected by me? | **No.** I obtained the original sequences from the third-party project below. I created the processed features/labels from that file. |

### **Dataset A planned analysis**

1. Report class frequencies and distributions of the numerical features, per source.  
2. Standardize the training-only predictors; examine correlations and PCA explained variance.  
3. Fit Logistic Regression, SVM, or Random Forest; compare with and without PCA.  
4. Report confusion matrices, accuracy, per-class precision/recall, macro-F1, and computation time.  
5. Interpret what distinguishes source groups without equating source identification with entropy.

### **Dataset B planned analysis**

1. Measure the empirical proportion of 1s and marginal entropy separately by source.  
2. Compare a **source-specific constant-probability baseline** with Logistic Regression, Random Forest, or an MLP trained on preceding bits.  
3. Evaluate held-out accuracy, precision, recall, F1, **predictive cross-entropy/log loss**, and calibration. Compare PCA/no-PCA variants if informative.  
4. Test whether models improve on source-specific frequency baselines; report uncertainty and avoid overinterpreting small gains.  
5. Compare next-bit predictive cross-entropy with marginal entropy; consider established statistical and NIST SP 800-90B methods separately when data acquisition conditions permit.

## **5\. Why this is an engineering problem (course justification)**

This is an engineering problem rather than only an information-theory exercise because it requires **designing, implementing, validating, and comparing a computational measurement-analysis system** for an important engineered component: a random-number generator. RNG performance is relevant to hardware security, quantum circuits, authentication, and secure communication. Finite measurements can show source-specific bias or temporal structure, and efficient automated diagnostics are useful when processing large volumes of measurements.

The system has **defined inputs** (binary sequences), **engineered predictors** (statistical features or preceding bits), **defined outputs** (generator-source or next-bit predictions), and **quantitative design objectives** (accuracy/F1, sensitivity to predictability, PCA dimensionality, and model runtime). Comparing such models and their limitations is an engineering trade-off study.

**Scope:** This project tests whether predictive modeling detects statistical structure in this dataset. It does **not** claim to implement a certified RNG entropy-estimation procedure, to prove/defeat cryptographic security, or to identify physical devices absent verified ground truth.

### **Original source, acquisition, and provenance**

* **Direct dataset URL (downloaded file):** [https\://github.com/hubertkolcz/NoiseVsRandomness/blob/main/AI\_2qubits\_training\_data.txt](https://github.com/hubertkolcz/NoiseVsRandomness/blob/main/AI_2qubits_training_data.txt)  
* **Direct raw-text URL:** [https\://raw.githubusercontent.com/hubertkolcz/NoiseVsRandomness/main/AI\_2qubits\_training\_data.txt](https://raw.githubusercontent.com/hubertkolcz/NoiseVsRandomness/main/AI_2qubits_training_data.txt)  
* **Repository supplying the file:** [https\://github.com/hubertkolcz/NoiseVsRandomness](https://github.com/hubertkolcz/NoiseVsRandomness)  
* **Upstream fork containing this dataset:** [https\://github.com/anya-kondamani/NoiseVsRandomness](https://github.com/anya-kondamani/NoiseVsRandomness)  
* **Earlier project ancestry:** [https\://github.com/pandey-tushar/Y\_Quantum](https://github.com/pandey-tushar/Y_Quantum)  
* **Underlying QRNG protocol referenced by the research repository:** [https\://github.com/dorahacksglobal/quantum-randomness-generator](https://github.com/dorahacksglobal/quantum-randomness-generator)

