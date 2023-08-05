Identifying And Analyzing Face Aging in Alzheimer's Patients using Deep Learning Methods
---

- [Introduction](#introduction)
- [Materials and Method](#materials-and-method)
- [Proposed Method](#proposed-method)
- [Results](#results)
- [Reference](#reference)



Introduction
---

The current methods for estimating age using facial images are mainly divided into regression methods and multi-task classification methods. Regression methods aim to find the relationship between facial features and age and directly output the predicted age value. On the other hand, multi-task classification methods aim to produce probabilities representing each age category, and the final age group is determined by comparing these probability values. The output of multi-task classification methods is discrete classes.

Currently popular regression methods include Support Vector Regression (SVR)[[1]](#reference), Partial Least Squares (PLS)[[2]](#reference), Kernel Partial Least Squares (KLPS)[[3]](#reference), Canonical Correlation Analysis (CCA)[[4]](#reference), and Orthogonal Gaussian Process (OGP) regression[[5]](#reference). However, the aging process of facial features is a non-stationary process, meaning that the degree of aging in a person's face varies with age. For example, the facial aging of a teenager will be more significant than that of an elderly person, even if they both age by one year. As mentioned in [[6]](#reference), traditional regressors tend to overfit when learning non-stationary kernels from training datasets. To address age prediction with such characteristics, ordinal regression or ordinal classification methods have been derived.

Ordinal regression is commonly used to predict labels with ordinal data, where classes have an inherent order. Many age estimation ordinal regression algorithms use classification algorithms. They consider not only the numerical age values corresponding to age labels but also take into account the ordinal property of age, where adjacent age categories have a meaningful order relationship. By considering the ordinal nature of age labels, the methods can provide more reliable information for facial age estimation compared to treating them as merely distinct categories.

To effectively leverage well-studied classification algorithms for handling ordinal targets in conventional Convolutional Neural Network (CNN) classification methods, researchers have simplified the problem into multiple binary classification tasks[[11]](#reference)[[12]](#reference). Niu et al. proposed the Ordinal Regression CNN (OR-CNN)[[13]](#reference) method in 2016, following the approach outlined in [[11]](#reference).

In OR-CNN, the ordinal regression problem with K ranked classes is divided into K-1 binary classification tasks. The process is illustrated in Figure 1. For instance, when the model needs to predict the age of a person with a true age of 40 years, it performs multiple binary classifications. Each binary classification task aims to determine whether the input facial age is greater than the current rank level.

The model sequentially checks if the input image's facial age is greater than rank 10, then rank 11, and so on, up to rank 50. If the predicted value is not greater than the current rank level, it means the predicted age is within that rank. On the other hand, if the prediction is greater than the current rank level, the model proceeds to the next rank's binary classification until the input facial age is smaller than the current rank level. The final predicted age is determined based on the highest rank level for which the input facial age is predicted to be greater.

However, the OR-CNN method does not guarantee prediction consistency, meaning that the neural network's age estimation probabilities do not decrease consistently with increasing rank levels. For example, when predicting the age of a person with a true age of 50 years, the probability of predicting their age to be greater than 20 years may be smaller than the probability of predicting it to be greater than 30 years. However, since age is an ordered data with the assumption that everyone ages at the same rate, the neural network should ensure that the probability of predicting an age to be 20 years is greater than the probability of predicting it to be 30 years. The probability of age estimation should monotonically decrease with increasing age, as illustrated in Figure 2.

To ensure rank monotonicity and improve the accuracy of age prediction in this study, we reference the method proposed by Cao et al. in 2020, called COnsistent RAnk Logits CNN (CORAL-CNN) [[14]](#reference). This method addresses the issue of inconsistency in the classifier's predictions in OR-CNN. CORAL-CNN employs ResNet-34 (He et al., 2016) [[15]], a powerful architecture for image classification tasks, as its main model structure. The output layer of ResNet-34 is replaced with the corresponding K-1 binary classification tasks. These K-1 binary tasks share the same weight values but have independent bias units to achieve rank monotonicity and ensure the uniformity of the classifiers.

By doing so, CORAL-CNN ensures that it produces predictions with rank consistency, addressing the rank inconsistency problem encountered in OR-CNN and improving the prediction accuracy. Therefore, in this study, to achieve better accuracy in predicting age, which is an ordered data type, we adopt CORAL-CNN as the tool to estimate ages from facial images.


<p align = "center" >

<img width="600" src="https://github.com/fefei69/Alzheimer-Patients-face-aging-analysis/blob/master/img/1.jpg"/>

</p>


Materials and Method
---
In the CORAL-CNN method, three datasets were used to test the model's performance. These datasets are as follows:

1.MORPH-2 dataset (Ricanek and Tesafaye, 2006) [[16]](#reference): It contains a total of 55,608 facial images, with ages ranging from 16 to 70 years.

2.CACD dataset (Chen et al., 2014) [[17]](#reference): This dataset consists of 159,449 facial images, with ages ranging from 14 to 62 years.

3.Asian Face Database (AFAD) by Niu et al. (2016) [[13]](#reference): It contains 165,501 facial images of Asian individuals, with ages ranging from 15 to 40 years.

From Table 1, it can be observed that the CORAL-CNN model achieved the best accuracy when trained with the MORPH-2 dataset. However, the focus of this research is on elderly individuals, and the age range of the majority of data in the dataset significantly exceeds that of the MORPH-2, CACD, and AFAD datasets. The mean age in the family dataset is 75.02 years, and in the patient dataset, it is 80.68 years (as shown in Table 2).

Due to this age difference, using the CORAL-CNN model trained on these three datasets to predict the age of elderly individuals in this study will result in significant errors. One issue is that the neural network cannot output values beyond the age range of the training dataset. For instance, using the MORPH-2 dataset as an example, the maximum age in this dataset is 70 years. Hence, even if the neural network is trained on this dataset, it can only output ages up to 70 years. This limitation leads to intrinsic errors when predicting data with ages greater than 70 years.

Furthermore, the neural network has not been trained on data above 70 years, making it difficult to distinguish the differences between ages above 70 years. This results in substantial errors when predicting the age of individuals in the family and patient datasets in this study.

Therefore, directly applying a pre-trained model to this study's dataset is not suitable, and additional efforts are needed to address the specific age range and characteristics of the elderly individuals in this research.



<div align="center">
    
| **Datasets** | **Age Range** | **Image Resolution** |     **MAE/RMSE**      |
| ------------ |:-------------:|:--------------------:|:---------------------:|
| **MORPH-2**  |     16-70     |         HIgh         | 2.64±0.02 / 3.65±0.04 |
| **CACD**     |     14-62     |        Medium        | 3.47±0.05 / 4.71±0.06 |
| **AFAD**     |     15-40     |         Low          | 5.25±0.01 / 7.48±0.06 |

</div>

<p align="center">Table 1. comparison of different datasets</p>

<div align="center">
    
|              | **Number** | **AVG±STD** | **Age Range** |
| ------------ |:----------:|:-----------:|:-------------:|
| **Family**   |    253     | 75.02±7.34  |     53-92     |
| **Patients** |    1722    | 80.68±6.77  |     58-99     |
| **UTKface**  |    4429    | 64.82±10.92 |     52-96     |
    
</div>

<p align="center">Table 2. Informations of our dataset</p>


Proposed Method
---
The objective of this research is to verify whether Alzheimer's disease patients show accelerated aging in their visual appearance, meaning that their facial visual age may appear older than their actual age. Therefore, we will separately estimate the ages of Alzheimer's disease patients and non-affected family members. We will analyze the differences in age estimation errors between the two datasets. However, as mentioned earlier, the age distributions in the three datasets used in the CORAL-CNN method (MORPH-2, CACD, and AFAD) are lower than those in the Alzheimer's disease patient and family datasets. Hence, we cannot directly use models trained on these three datasets to predict the ages of our dataset.

In typical image classification tasks requiring the transfer of a model from a source domain to a similar target domain with a smaller dataset, the Transfer Learning technique is often used. It involves freezing most of the neural network's front layers and training only the non-frozen layers on the target domain dataset. In this research, we theoretically could have employed the Transfer Learning technique by freezing the early layers of CORAL-CNN and training the last few layers with the family dataset to adapt the pre-trained CORAL-CNN model to the target domain of family members (unseen test set) and patients, as shown in Figure 5. This would enable the tuned model to perform well in predicting the ages of family members and patients in the higher age range. However, the target domain for this research, the family dataset, contains too few data points (only 253 samples), making it susceptible to overfitting during training. Overfitting would cause the model to memorize the training data and perform poorly on unseen test data, resulting in large prediction errors.

To address this issue, we used the UTKface dataset [18], which contains a large number of facial images with age distributions ranging from 0 to 116 years. We extracted a subset of the UTKface dataset with ages ranging from 52 to 96 years, aligning more closely with the age distributions in the family and patient datasets (52 to 99 years). This subset contained 4429 facial images with a mean age of 64.82 years (as shown in Table 2).

For model training, we split the 4429 UTKface samples into 3543 training samples, 443 validation samples, and 443 test samples. We used the CORAL-CNN architecture and trained the model from scratch, observing significant improvements in Mean Absolute Error (MAE) for training, testing, and validation sets when we increased the input size to 224x224 pixels. This improvement might be attributed to the less pronounced facial features of elderly individuals, such as wrinkles, which reduce the differences in facial age among elderly individuals compared to younger individuals, as shown in Figure 6. As a result, the neural network faces greater difficulty in distinguishing between two elderly individuals in their 70s and 80s than between two young individuals in their 10s and 20s. Thus, CORAL-CNN had difficulty training a suitable model when the input size was 120x120 pixels. However, when the input size increased to 224x224 pixels, the model learned the subtle facial features of elderly individuals.

Regarding the parameter settings for model training, most of the parameters were the same as in the CORAL-CNN method, with a learning rate α = 5×10-5, the Adam optimizer [19], and 200 epochs.

Since the subset of UTKface dataset (4429 samples) used in our study is significantly smaller than the datasets used in CORAL-CNN (MORPH-2, CACD, and AFAD)[14], there is a risk of overfitting due to the limited data size during training. To assess the model's generalization capability, we analyzed the relationship between the true age and the prediction errors on the test and training sets (as shown in Figure 8). The ideal scenario is that the regression lines fitted to the test and training datasets are close to horizontal, indicating that the true age and the model's prediction errors are unrelated. When a linear trend exists between the true age and the prediction errors (as shown by the red line in Figure 8), it indicates a certain level of association between the true age and the prediction errors, implying that the model has not learned well. If the regression line slopes toward -1, it suggests that the model tends to output the same value for different ages, indicating severe overfitting. Interestingly, when analyzing CORAL-CNN's pre-trained model on the CACD dataset, we found that the model's test dataset also exhibited a linear trend similar to our model. This suggests that the linear trend might be an issue inherent in using classification-based methods for age estimation.

Results
---
In this research, we addressed the age estimation problem by analyzing the relationship between true age and prediction errors on the test dataset, which allowed us to observe a linear trend between them. Therefore, we anticipated that this linear trend would also appear when the model predicts unseen data. To address this, we proposed a method to correct the prediction errors using a linear regressor. For each individual's age prediction $ŷ_i$, true age $y_i$, and prediction error $e_i = ŷ_i - y_i$, we established a linear equation $e_i = a * y_i + b$, where a and b are constants. This equation allowed us to predict the prediction error $e_i$ when the true age is $y_i$. We then adjusted (predicted) the remaining 23 samples of family members and 1722 samples of patients using this linear equation to make the prediction errors closer to zero, eliminating the correlation between the true age and the prediction errors. The adjusted prediction $ŷ̂_i$ was obtained as $ŷ_i - (a * y_i + b)$. Finally, we analyzed the relationship between the new prediction errors $ê_i = ŷ̂_i - y_i$ and the true ages $y_i$ to obtain a non-correlated horizontal fitting line, making the predictions more reliable.

To ensure that the fitting equation is not a result of specific data combinations and to minimize the prediction bias caused by the limited data, we conducted 100 rounds of fitting and correction. For each round, we randomly selected 230 samples from the family dataset for fitting and applied the correction on the remaining 23 samples of family members and 1722 samples of patients. We then averaged the corrected prediction errors for each sample over nine times for family members (i.e., each sample was selected on average 23/253 * 100 times) and 100 times for patients. We analyzed the relationship between the corrected prediction errors and the true ages (as shown in Figure 10). The fitting line for the family dataset (in red) was close to horizontal, indicating that the corrected family dataset successfully eliminated the correlation between the prediction errors and the true ages, demonstrating the representativeness of our algorithm (as shown in Figure 7) for predicting the facial ages of healthy elderly individuals. In contrast, the fitting line for the patient dataset still exhibited a linear trend, but this trend was much milder after correction. The slope of the fitting line for the patient dataset decreased from -0.461 to -0.084 after correction. It is worth mentioning that our research aims to explore whether patients show accelerated facial aging compared to healthy individuals (family members). Thus, the difference in prediction results between patients and family members is the focus of our analysis. As previously mentioned, the CORAL-CNN architecture is more adept at learning age classification problems in lower age groups (e.g., 60-69 years) compared to higher age groups (e.g., 90-99 years). Therefore, the linear trend observed in the patient data after correction (as shown by the red line in Figure 10) indicates that the model tends to overestimate the ages of patients with lower actual ages and underestimate the ages of patients with higher actual ages. This finding confirms the phenomenon of accelerated visual aging in Alzheimer's disease patients. The model struggles to distinguish the visual age differences among elderly individuals, leading it to output the average value for different age groups and causing the prediction errors to become negative for older ages. Additionally, we performed a Hypothesis T-Test by averaging all the corrected prediction errors for family members and patients to test if the patients' prediction errors were significantly greater than those of family members. The null hypothesis $H_0: μ_{Pt} - μ_{Fm} > 0$ and the alternative hypothesis $H_1: μ_{Pt} - μ_{Fm} ≤ 0$. The P-value obtained was 0.117 (two-tailed), indicating that the patients' prediction errors were statistically larger than those of family members, further supporting the existence of accelerated facial aging in Alzheimer's disease patients compared to healthy individuals.



**Reference**
---
[1] G. Guo, G. Mu, Y. Fu, and T. Huang. Human age estimation using bio-inspired features. CVPR, pages 112–119, 2009.

[2] G. Guo and G. Mu. Simultaneous dimensionality reduction
and human age estimation via kernel partial least squares regression.
CVPR, pages 657–664, 2011

[3] Guo, G., Mu, G.: Simultaneous dimensionality reduction
and human age estimation via kernel partial least squares
regression. In: Computer vision and pattern recognition
(cvpr), 2011 ieee conference on, pp. 657–664. IEEE (2011)

[4] G. Guo and G. Mu. Joint estimation of age, gender and ethnicity:
Cca vs. pls. FG, pages 1–6, 2013.

[5] Zhu, K., Gong, D., Li, Z., Tang, X.: Orthogonal gaussian
process for automatic age estimation. In: Proceedings of
the 22nd ACM international conference on Multimedia,
pp. 857–860. ACM (2014)

[6] K. Chang, C. Chen, and Y. Hung. Ordinal hyperplanes ranker
with cost sensitivities for age estimation. CVPR, pages 585–
592, 2011.

[7] R. Herbrich, T. Graepel, and K. Obermayer. support vector
learning for ordinal regression. Proc. Int. Conf. Artif. Neural
Netw, pages 97–102, 1999.

[8] K. Crammer and Y. Singer. Pranking with ranking. NIPS,
pages 641–647, 2002.

[9] A. Shashua and A. Levin. Ranking with large margin principle:
Two approaches. NIPS, pages 961–968, 2003.

[10] Levi, G., Hassner, T., 2015. Age and gender classification using convolutional
neural networks, in: IEEE Conference on Computer Vision and Pattern
Recognition Workshops, pp. 34–42.
[11] L. Li and H. Lin. Ordinal regression by extended binary
classification. NIPS, pages 865–872, 2006.

[12] E. Frank and M. Hall. A simple approach to ordinal classification.
Lecture Notes in Artificial Intelligence, pages 145–
156, 2001.

[13] Niu, Z., Zhou, M., Wang, L., Gao, X., Hua, G., 2016. Ordinal regression with
multiple output cnn for age estimation, in: IEEE Conference on Computer
Vision and Pattern Recognition, pp. 4920–4928.


[14] Wenzhi Cao, Vahid Mirjalili, Sebastian Raschka (2020): Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation. Pattern Recognition Letters.

[15] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image
recognition, in: IEEE Conference on Computer Vision and Pattern Recognition,
pp. 770–778.

[16] Ricanek, K., Tesafaye, T., 2006. Morph: A longitudinal image database of
normal adult age-progression, in: IEEE Conference on Automatic Face and
Gesture Recognition, pp. 341–345.

[17] Bor-Chun Chen, Chu-Song Chen, andWinston H. Hsu. Face
recognition and retrieval using cross-age reference coding
with cross-age celebrity dataset. IEEE Trans. Multimedia,
17:804–815, 2015. 5, 6

[18] Zhifei Zhang, Yang Song, and Hairong Qi. Age progression/
regression by conditional adversarial autoencoder. In
CVPR, 2017b. 5, 6

[19] Kingma, D.P., Ba, J., 2015. Adam: A method for stochastic optimization,
in: Bengio, Y., LeCun, Y. (Eds.), 3rd International Conference on Learning
Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference
Track Proceedings, pp. 1–8. URL: http://arxiv.org/abs/1412.6980.

