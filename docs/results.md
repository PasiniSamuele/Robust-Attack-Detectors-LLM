<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Results

## Introduction

In our application scenarios (XSS/SQLi detection), we give more weight to false negatives (resulting in low recall) with respect to false positives (resulting in low precision), since a non-detected attack can cause much more damage than a benign request detected as an attack.
For this reason, the main performance metric used in our empirical study is the F2-Score, referred to as F2 for simplicity, which gives double importance to recall than to precision: $F2 = \frac{5 * precision * recall}{4 * precision + recall}$.
To mitigate a possible threat to the construct validity associated with the choice of F2-Score as the performance metric, we replicated all experiments using accuracy as an alternative performance metric. For space reasons, we could not report the results obtained with both metrics in the paper, so here, where we do not have space limitations you can find the results obtained with both metrics.

## RQ1
How helpful is RAG in generating better security-critical functions? How does it perform when combined with Few-shot examples?



### F2-Score
Figure below illustrates the impact of RAG on function generation for XSS and SQLi in the NED (No Evaluation Dataset) scenario, showing the F2 differences between the configurations with and without RAG, across all possible configurations. Each bar represents the average F2 score difference for a specific Model-Temperature pair when using RAG, compared to the same pair without RAG. Results indicate that employing RAG generally enhances the performance of function generation for both tasks. The number of Model-Temperature pairs benefiting from RAG is much larger than the number of pairs showing degradation, and the improvements are statistically significant, as evidenced by p-values ( $\approx 10^{-66}$ for XSS and $\approx 10^{-24}$ for SQLi) below the standard threshold of 0.05. 


<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/rag_improvement_hist_xss_f2.png" alt="Image 1" style="width: 45%; margin: 5px;">
            <img src="../plots/rag_improvement_hist_sqli_f2.png" alt="Image 2" style="width: 45%; margin: 5px;">
        </div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-top: 10px;">
            Figure: Difference between the F2 of Generated Function Runs with RAG (i.e., RAG Prompt and RAG-Few-shot Prompt) and the F2 of Generated Function Runs without RAG (i.e., Basic Prompt and Few-shot Prompt), for XSS detection (left) and SQLi detection (right).
        </div>
    </div>
</body>
</html>

---

We further investigate the benefit of combining Few-shot examples with RAG using a similar setting as Figure above. Figure below shows that
while the addition of Few-shot examples shows some benefits for SQLi, the same cannot be said for XSS. These findings suggest that the usage of Few-shot examples may not always provide advantages when RAG is already employed, indicating that in the NED (No Evaluation Dataset) scenario, it could be preferable to omit them.

<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/few_shot_improvement_xss_f2.png" alt="Image 1" style="width: 45%; margin: 5px;">
            <img src="../plots/few_shot_improvement_sqli_f2.png" alt="Image 2" style="width: 45%; margin: 5px;">
        </div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-top: 10px;">
            Figure: Difference between the F2 of Generated Function Runs with RAG Few-shot Prompt and the F2 of Generated Function Runs with RAG Zero-shot Prompt, for XSS detection (left) and SQLi detection (right).
        </div>
    </div>
</body>
</html>

### Accuracy

The figure below demonstrates the impact of RAG on function generation for XSS and SQLi in the NED (No Evaluation Dataset) scenario. It illustrates the differences in accuracy between configurations with and without RAG across all possible configurations. Each bar represents the average accuracy score difference for a specific Model-Temperature pair when using RAG, compared to the same pair without RAG. The results show that using RAG generally improves function generation performance for both tasks. The number of Model-Temperature pairs that benefit from RAG is significantly higher than those that do not, and these improvements are statistically significant.TODO pvalue.

<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/rag_improvement_hist_xss_acc.png" alt="Image 1" style="width: 45%; margin: 5px;">
            <img src="../plots/rag_improvement_hist_sqli_acc.png" alt="Image 2" style="width: 45%; margin: 5px;">
        </div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-top: 10px;">
            Figure: Difference between the Accuracy of Generated Function Runs with RAG (i.e., RAG Prompt and RAG-Few-shot Prompt) and the Accuracy of Generated Function Runs without RAG (i.e., Basic Prompt and Few-shot Prompt), for XSS detection (left) and SQLi detection (right). 
        </div>
    </div>
</body>
</html>

---

We further investigate the benefit of combining Few-shot examples with RAG using a similar setting as Figure above. Figure below shows that the usage of Few-shot examples may not always provide advantages when RAG is already employed. 

<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/few_shot_improvement_xss_acc.png" alt="Image 1" style="width: 45%; margin: 5px;">
            <img src="../plots/few_shot_improvement_sqli_acc.png" alt="Image 2" style="width: 45%; margin: 5px;">
        </div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-top: 10px;">
            Figure: Difference between the Accuracy of Generated Function Runs with RAG Few-shot Prompt and the Accuracy of Generated Function Runs with RAG Zero-shot Prompt, for XSS detection (left) and SQLi detection (right). 
        </div>
    </div>
</body>
</html>


## RQ2
Is the selection of
top_k functions via Self-Ranking an effective strategy for
enhancing the performance of security-critical functions?



### Top_k F2-Score
 The violin plots shown in Figure below depict the effect of Self-Ranking, i.e., $top\_k$ selection across three values of $k$ for the two tasks, considering all the possible pairs of function configurations and synthetic dataset generation. Overall, we observe a clear improvement, particularly for XSS. The region between quartiles for XSS falls largely between a 20\%pt and 40\%pt improvement, with an average improvement of 25\%pt and an improvement that affects 98\% of the cases. While the improvement for SQLi is less pronounced, we still observe improvements in 73\% cases, with an average improvement of 4\%pt. Additionally, we can see that, as $k$ increases, the average improvement decreases for both tasks, but the gains become more stable.
The improvements given by the usage of $top\_k$ selection are statistically significant: the p-values obtained with the Wilcoxon signed-rank test are below the threshold of 0.05 for both the tasks and all the values of k.

In the EDA scenario, we observe that utilizing the Self-Ranking mechanism to select the top\_k functions is more effective than not employing it: when compared to solely using $U_{best}$, employing $S_{best}$ achieves 3.21\%pt and 4.94\%pt increases in F2, for XSS and SQLi respectively.

<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/rq2_top_1_f2_xss.png" alt="Image 1" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_3_f2_xss.png" alt="Image 2" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_5_f2_xss.png" alt="Image 3" style="width: 30%; margin: 5px;">
        </div>
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/rq2_top_1_f2_sqli.png" alt="Image 4" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_3_f2_sqli.png" alt="Image 5" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_5_f2_sqli.png" alt="Image 6" style="width: 30%; margin: 5px;">
        </div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-top: 10px;">
Figure: F2 given by top_k selection (i.e., Self-Ranking) for XSS detection (first row) and SQLi detection (second row), with k=1, k=3 and k=5.        </div>
    </div>
</body>
</html>

### Top_k accuracy
The violin plots shown in Figure below depict the effect of Self-Ranking, i.e., $top\_k$ selection across three values of $k$ for the two tasks, considering all the possible pairs of function configurations and synthetic dataset generation. 

TODO HERE ADD DESCRIPTION

<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/rq2_top_1_acc_xss.png" alt="Image 1" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_3_acc_xss.png" alt="Image 2" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_5_acc_xss.png" alt="Image 3" style="width: 30%; margin: 5px;">
        </div>
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <img src="../plots/rq2_top_1_acc_sqli.png" alt="Image 4" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_3_acc_sqli.png" alt="Image 5" style="width: 30%; margin: 5px;">
            <img src="../plots/rq2_top_5_acc_sqli.png" alt="Image 6" style="width: 30%; margin: 5px;">
        </div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-top: 10px;">
Figure: Accuracy given by top_k selection (i.e., Self-Ranking) for XSS detection (first row) and SQLi detection (second row), with k=1, k=3 and k=5.        </div>
    </div>
</body>
</html>

## RQ3
Do the functions generated by LLMs, when assessed on an existing evaluation dataset, demonstrate comparable performance to state-of-the-art ML models trained specifically for the task?



### Comparison with SOTA (F2-Score)
To compare ours with learning-based SOTA techniques, we select $U^{best}$ and $top\_k$ functions based on $val\_set$ using F2-Score ( $M = F_2$ ). For XSS, $U^{best}$ is $(GPT-4T, 0.0, 10, T)$, while for SQLi, it is $(GPT-4T, 0.0, 2, T)$. We establish a baseline to better understand the improvement offered by adopting our approach. The baseline is obtained using Basic Prompt, without Few-shot examples and RAG.


Table below presents the results of the comparison. We observe a significant improvement of 25\%pt and 18\%pt for XSS and SQLi respectively, when compared to the baseline. 
There is a slight decrease in F2 compared to SOTA models, with a 3.15\%pt and 0.83\%pt drop for XSS and SQLi, respectively. 
We argue that the slight performance gap between our approach and SOTA models is understandable, given our approach's training-free nature and direct applicability to multiple tasks. <!--These results provide empirical support for our claim that incorporating external knowledge and Self-Ranking is essential for LLMs to achieving competitive performance with SOTA models. -->


<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="../tables_screenshot/rq3_f2.png" alt="Descriptive Alt Text" style="width: 65%; margin: 5px;">
    </div>
</body>
</html>


### Comparison with SOTA (Accuracy)
In this section we select the $U^{best}$ and $top\_k$ functions based on the validation set using the Accuracy ( $M = Accuracy$). For XSS, the optimal configuration $U^{best}$ is $(Opus, 0.0, 6, F)$, while for SQLi, it is $(GPT-4, 0.0, 2, T)$. To establish a reference point, we use a baseline derived from the Basic Prompt, which lacks Few-shot examples and Retrieval-Augmented Generation (RAG).

The table below shows the results of our comparison. Our method achieves a substantial improvement of 21%pt for XSS and 18%pt for SQLi when compared to the baseline. However, there is a minor decrease in the Accuracy relative to SOTA models, with a reduction of 0.3 percentage points for XSS and 1.33 percentage points for SQLi. We believe that this small performance gap is justified, considering our method's training-free nature and its direct applicability to various tasks.

<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="../tables_screenshot/rq3.png" alt="Descriptive Alt Text" style="width: 65%; margin: 5px;">
    </div>
</body>
</html>

## RQ4
Can the optimal parameters for function and synthetic data generation in one task be transferred and applied to achieve effective results in other tasks?


### Transferability F2-Score
Table below shows the best configuration of each task, specifically for function generation ($U^{best}$) and synthetic dataset generation ($S^{best}$), across different $k$ values with $\mathcal{M} = F_2$. In our notation, Task 1 is XSS and Task 2 is SQLi, and $A \rightarrow B$ represents the transfer of a configuration from one task to another. As an illustration, in $U^{best}_1 \rightarrow U^{transf}_2$, $U^{best}_1$ denotes the best configuration for Task 1, whereas $U^{transf}_2$ represents the transferred configuration, which originates from Task 1 ($U^{best}_1$) and is subsequently  evaluated on Task 2.



<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="../tables_screenshot/rq4_ubest_f2.png" alt="Descriptive Alt Text" style="width: 65%; margin: 5px;">
    </div>
</body>
</html>

Table below presents the transferability results. The $F2(U^{best}, S^{best})$ columns indicate F2 computed on the original task with its best configuration, "Avg. F2" columns represent the average F2 computed across all the $U-S$ pairs for a given $k$, and the $F2(U^{transf}, S^{transf})$ columns show F2 computed using transferred configurations. Comparing the transferred configuration's performance to the average column provides a good estimate of the benefits of configuration transfer over a mere random selection of a configuration for a new, unseen task.

The results support the effectiveness of transferring configurations. While there is a slight degradation in F2 compared to the original, best configuration (on average, 3\%pt for XSS and 8\%pt for SQLi ), we can observe those results of transferred configurations outperform the average F2, achieving on average, 16\%pt improvement for XSS and 10\%pt improvement for SQLi.


<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="../tables_screenshot/rq4_f2.png" alt="Descriptive Alt Text" style="width: 65%; margin: 5px;">
    </div>
</body>
</html>

### Transferability Accuracy
The table below highlights the optimal configurations for each task, specifically for function generation ( $U^{best}$ ) and synthetic dataset generation ( $S^{best}$ ), across various $k$ values, with $M = Accuracy$. Here, Task 1 refers to XSS and Task 2 to SQLi, with $A \rightarrow B$ indicating the transfer of a configuration from one task to another. For instance, in $U^{best}_1 \rightarrow U^{transf}_2$, $U^{best}_1$ denotes the optimal configuration for Task 1, while $U^{transf}_2$ represents this configuration transferred and evaluated on Task 2.


<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="../tables_screenshot/rq4_ubest.png" alt="Descriptive Alt Text" style="width: 65%; margin: 5px;">
    </div>
</body>
</html>

The table below presents the results of these transferability tests. The $Accuracy(U^{best}, S^{best})$ columns display the Accuracy achieved on the original task with its best configuration. The "Avg. Acc." columns show the average Accuracy across all $U-S$ pairs for a given $k$, and the $Acc(U^{transf}, S^{transf})$ columns demonstrate the Accuracy using transferred configurations. Comparing the performance of transferred configurations to the average column offers insight into the benefits of transferring configurations versus randomly selecting a configuration for a new, unseen task.

The results validate the efficacy of configuration transfer. Although there is a slight decrease in Accuracy compared to the original best configuration (on average, 9.3 percentage points for XSS and 7.8 percentage points for SQLi), the transferred configurations generally surpass the average Accuracy, yielding an average improvement of 4.2 percentage points for XSS and 9.4 percentage points for SQLi.


<!DOCTYPE html>
<html lang="en">
<body>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="../tables_screenshot/rq4.png" alt="Descriptive Alt Text" style="width: 65%; margin: 5px;">
    </div>
</body>
</html>