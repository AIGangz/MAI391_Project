import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1779)

# Configuration
#st.set_page_config(layout="wide")

# Side bar
LIST_CONTENTS = [
    "1. PCA là gì?",
    "2. Các bước thực hiện phương pháp PCA",
    "3. Trực quan hoá dữ liệu",
    "4. Xây dựng model sử dụng PCA"
]

st.sidebar.title('Mục lục')
slbContent = st.sidebar.selectbox(
    "Chọn phần bạn muốn xem",
    LIST_CONTENTS
)


# Subpages
st.markdown('''
# 1. What is PCA?

**PCA (Principal Component Analysis)** is a method of computing the principal components
and using them to perform a change of basis on the data, or reduce the dimension of dataset.

**Why do we need to reduce the dimension of the dataset?**
- It's hard to analyze, visualization is nearly impossible.
- Storage of data is really expensive.
- Many dimensions are redundant, which can be explained by combination of other dimensions or those dimensions hold too little information about dataset.    
''')

NUM_POINTS = 75
x = np.random.normal(0, 4, NUM_POINTS)
y = np.random.normal(0, 0.3, NUM_POINTS)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

ax[0].scatter(x, y)
ax[0].axhline(y=0, color='r')
ax[0].axvline(x=0, color='r')
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[0].grid(True, 'major')
ax[0].axis([-10,10,-10,10])
ax[0].set_title('(1.a) Original data')

ax[1].scatter(x, [0 for _ in range(NUM_POINTS)])
ax[1].axhline(y=0, color='r')
ax[1].axvline(x=0, color='r')
ax[1].grid(True, 'major')
ax[1].axis([-10,10,-10,10])
ax[1].set_title('(1.b) Project on x-axis')

ax[2].scatter([0 for _ in range(NUM_POINTS)], y)
ax[2].axhline(y=0, color='r')
ax[2].axvline(x=0, color='r')
ax[2].grid(True, 'major')
ax[2].axis([-10,10,-10,10])
ax[2].set_title('(1.c) Project on y-axis')

st.pyplot(fig)

st.markdown('''
*The above example shows that when data points are proejected on x-axis (1.b), they are more clearly dispersed than
y-axis (1.c)*
''')

st.markdown('''
# 2. Steps involved in the PCA
## 2.1. Step-by-Step instruction
**Step 1**: Standardize the dataset.

**Step 2**: Calculate the covariance matrix for the features in the dataset.

**Step 3**: Calculate the eigenvalues and eigenvectors for the covariance matrix.

**Step 4**: Sort eigenvalues and their corresponding eigenvectors.

**Step 5**: Pick `k` eigenvalues and form a matrix of eigenvectors.

**Step 6**: Transform the original matrix.
''')
input_number = st.number_input(label='Input number of random data points', min_value=2, max_value=500, value=5)
x1 = np.random.randint(0, 10, int(input_number))
x2 = np.random.randint(0, 10, int(input_number))
x3 = np.random.randint(0, 10, int(input_number))
ex_df = pd.DataFrame({'x1': x1,
'x2': x2,
'x3': x3}, index=range(1, input_number+1))



st.markdown('''
Given example dataset:
''')
st.table(ex_df)

fig, ax = plt.subplots(figsize=(20,8))
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, x3)
st.pyplot(fig)

st.markdown('''
**Standardlize the features:**

To standardlize the data, we need to calculate the mean and standard deviation
''')

st.latex(r'x_{new} = \frac{x-\mu}{\sigma}')

#ex_df.mean(axis=1)

mean_std_df = pd.concat([ex_df.mean(), ex_df.std(ddof=0)], axis=1)
mean_std_df.columns = ['mean', 'std']

standard_ex_df = (ex_df - ex_df.mean()) / ex_df.std(ddof=0)

st.table(mean_std_df.T)

st.markdown('Data after standardlize:')
st.table(standard_ex_df)

fig, ax = plt.subplots(figsize=(20,8))
ax = plt.axes(projection='3d')
ax.scatter3D(standard_ex_df['x1'], standard_ex_df['x2'], standard_ex_df['x3'])
st.pyplot(fig)

st.markdown('''
**Calculate covariance matrix:**
''')

st.latex(r'''
Cov(X, Y) = \frac{\Sigma(x_i - \bar x)(y_i - \bar y)}{N-1}
''')


cov_ex_df = standard_ex_df.cov(ddof=1)
st.table(cov_ex_df)

st.markdown('''
**Calculate eigenvalues and eigenvectors:**
''')

eigvalues, eigvectors = np.linalg.eig(cov_ex_df.to_numpy())

idx = eigvalues.argsort()[::-1]
eigvalues = eigvalues[idx]
eigvectors = eigvectors[:, idx]

st.code(f'''Eigenvalues: {str(eigvalues)}
Eigenvectors: 
{str(eigvectors)}
''')

fig, ax = plt.subplots(figsize=(20,8))
ax = plt.axes(projection='3d')
ax.scatter3D(standard_ex_df['x1'], standard_ex_df['x2'], standard_ex_df['x3'])
#ax.view_init(0, 0)

v1, v2, v3 = zip(*eigvectors)
#print(v1, v2, v3)
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='red')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='green')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='purple')

st.pyplot(fig)

st.markdown('''
**Pick first 2 components and transform data:**
''')


pca_df = pd.DataFrame(np.dot(standard_ex_df.to_numpy(), eigvectors[:, :2]),
                    index=standard_ex_df.index, columns=['nx' + str(i) for i in range(1, 2+1)])

st.table(pca_df)

fig, ax = plt.subplots(figsize=(5,5))
ax.grid(True, 'major')
ax.scatter(pca_df['nx1'], pca_df['nx2'])
ax.axvline(x=0, color='g')
ax.axhline(y=0, color='r')
ax.axis([-5,5,-5,5])
st.pyplot(fig)

st.markdown(r'''## 2.2. Explaination
We assume that there're a eigenvector $$b \in R^D$$ and a set of datapoint  $$A\in R^{D \times N}$$.

Project $$A$$ on vector $$b$$ using formula:
''')
st.latex(r'''proj_ba_i = \frac{b^T \cdot a_i}{||b||}b''')

st.markdown(r'''In this case, $$b$$ is normed vector. So that, $$||b||=1$$
''')

st.latex(r'''proj_ba_i = (b^T \cdot a_i)b''')
st.latex(r'''z_i=b^T \cdot a_i''')

st.image('images/1.png')

st.latex(r'''Var(Z) = Var(b^T \cdot A) = \frac{\Sigma (z_i - \bar z)^2}{N-1}
= \frac{\Sigma (b^T \cdot a_i)^2}{N-1} = \frac{1}{N-1}\Sigma (b^T \cdot a_i \cdot b^T \cdot a_i)
=\frac{1}{N-1}\Sigma (b^T \cdot a_i \cdot a_i^T \cdot b) \\
=b^T \frac{\Sigma (a_i \cdot a_i^T)}{N-1}b\\
=b^T\cdot Cov(A) \cdot b=b^TSb''')


st.markdown(r'''Our goal is find $$\underset{b}{argmax}(V)$$, subject to $$||b||^2=1$$ (Cause $$S$$ is symmetric). Using **Lagrange multipliers** to find $$b$$:''')

st.latex(r'''\mathcal{L}(b, \lambda)=b^TSb + \lambda(1-b^T \cdot b )''')
st.latex(r'''\implies \nabla\mathcal{L}(b, \lambda)=
\begin{bmatrix} 2b^TS - 2\lambda b^T \\
    1-b^T\cdot b
     \end{bmatrix}=0''')

st.markdown(r'''Focus on the first equation:''')

st.latex(r'''b^TS = \lambda b^T \implies S^Tb = \lambda b''')
st.markdown(r'''$$\lambda$$ as eigenvalue, $$b$$ as eigenvector''')

st.markdown("## 2.3. PCA and SVD")
st.markdown(r'''
The SVD of data matrix $$X = [x_1 \cdot\cdot\cdot x_N] \in R^{D \times N}$$:
''')
st.latex(r'X=U\Sigma V^T')

st.markdown(r'where $$U \in R^{D\times D}, V \in R^{N \times N}$$ are orthonormal matrices and $$\Sigma \in R^{D\times N}$$ is a matrix whose only none zero entries are singular values $$\sigma_{ii} >= 0$$. The data covariance matrix:')

st.latex(r'S=\frac{1}{N}XX^T=\frac{1}{N}U\Sigma V^T V \Sigma^T U^T = \frac{1}{N}U\Sigma\Sigma^T U^T')

st.markdown('The columns of $$U$$ are the eigenvectors of $$XX^T$$. The eigenvalues $$\lambda_d$$ of $$S$$ are related to the singular values of $$X$$:')

st.latex(r'\lambda_d = \frac{\sigma^2_d}{N}')

st.markdown("## 2.4. Power Iteration for Eigenvectors and Eigenvalues")

st.markdown(r'For large size matrices, this is not possible to find all eigenvalues and eigenvectors. So we using **power iteration** converges to the eigenvectors associated with the largest eigenvalue of $$S$$:')

st.markdown(r'Choose initial vector $$x_0$$ NOT in the null space of $$S$$ and follow:')
st.latex(r'x_{k+1}=\frac{Sx_k}{||Sx_k||}')
st.latex(r'\lambda_{k+1} = ||Sx_k||')
##################
st.markdown("# 3. Dataset [Returns to Education]")
##################
if st.button('Explore the dataset!'):
    df = pd.read_excel('wage.xlsx', 'wage')
    st.dataframe(df)

    st.markdown('''## 3.1. Description
    1. **id**:                   person identifier
    2. **nearc2**:               =1 if near 2 yr college, 1966
    3. **nearc4**:               =1 if near 4 yr college, 1966
    4. **educ**:                years of schooling, 1976
    5. **age**:                in years
    6. **fatheduc**:              father's schooling
    7. **motheduc**:               mother's schooling
    8. **weight**:               NLS sampling weight, 1976
    9. **momdad14**:              =1 if live with mom, dad at 14
    10. **sinmom14**:               =1 if with single mom at 14
    11. **step14**:               =1 if with step parent at 14
    12. **reg661**:                =1 for region 1, 1966
    13. **reg662**:                 =1 for region 2, 1966
    14. **reg663**:                 =1 for region 3, 1966
    15. **reg664**:                 =1 for region 4, 1966
    16. **reg665**:                =1 for region 5, 1966
    17. **reg666**:              =1 for region 6, 1966
    18. **reg667**:               =1 for region 7, 1966
    19. **reg668**:                 =1 for region 8, 1966
    20. **reg669**:                  =1 for region 9, 1966
    21. **south66**:                 =1 if in south in 1966
    22. **black**:                =1 if black
    23. **smsa**:                  =1 in in SMSA, 1976
    24. **south**:                 =1 if in south, 1976
    25. **smsa66**:                =1 if in SMSA, 1966
    26. **wage**:                hourly wage in cents, 1976
    27. **enroll**:                  =1 if enrolled in school, 1976
    28. **KWW**:                     knowledge world of work score
    29. **IQ**:                       IQ score
    30. **married**:                  =1 if married, 1976
    31. **libcrd14**:                 =1 if lib. card in home at 14
    32. **exper**:                    age - educ - 6
    33. **lwage**:                    log(wage)
    34. **expersq**:                  exper^2''')

    st.dataframe(df.describe())

    st.code('''
categorial_columns = np.array(['nearc2', 'nearc4', 'momdad14', 'sinmom14', 'step14', 'reg661', 'reg662', 'reg663', 'reg664', 'reg665', 'reg666', 'reg667', 'reg668', 'reg669', 'south66', 'black', 'smsa', 'south', 'smsa66', 'enroll', 'married', 'libcrd14'])
quantitive_columns = np.array(['educ', 'age', 'fatheduc', 'motheduc', 'weight', 'KWW', 'IQ', 'expersq'])

x_columns = np.append(categorial_columns, quantitive_columns)
y_columns = 'lwage'
    ''')

    categorial_columns = np.array(['nearc2', 'nearc4', 'momdad14', 'sinmom14', 'step14', 'reg661', 'reg662',
        'reg663', 'reg664', 'reg665', 'reg666', 'reg667', 'reg668', 'reg669',
        'south66', 'black', 'smsa', 'south', 'smsa66', 'enroll', 'married', 'libcrd14'])
    quantitive_columns = np.array(['educ', 'age', 'fatheduc', 'motheduc', 'weight', 'KWW', 'IQ', 'expersq'])

    x_columns = np.append(categorial_columns, quantitive_columns)
    y_columns = 'lwage'

    st.markdown('''## 3.2. Analyzation''')
    st.markdown('''### 3.2.1. Checking null values''')
    st.code(''' 
    nan_values = df.isnull()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns]
    for i in columns_with_nan:
        print(i, df[i].isnull().sum())
    ########
    # fatheduc 690
    # motheduc 353
    # KWW 47
    # IQ 949
    # married 7
    # libcrd14 13
    ''')

    st.markdown('''### 3.2.2. Plotting continuous data''')
    st.code(''' 
    plt.figure(figsize=(18,18))
    for i in range(1, len(quantitive_columns)):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=quantitive_columns[i], bins=10)

    plt.subplot(3,3, i+1)
    sns.histplot(data=df, x='wage', bins=10)
    plt.subplot(3,3, i+2)
    sns.histplot(data=df, x='lwage', bins=10)
    plt.show()
    ''')

    fig = plt.figure(figsize=(18,18))
    for i in range(1, len(quantitive_columns)):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=quantitive_columns[i], bins=10)

    plt.subplot(3,3, i+1)
    sns.histplot(data=df, x='wage', bins=10)
    plt.subplot(3,3, i+2)
    sns.histplot(data=df, x='lwage', bins=10)
    st.pyplot(fig)

    st.markdown('''### 3.2.2. Plotting discrete data''')

    st.code('''
    plt.figure(figsize=(25,20))
    for i in range(1, len(categorial_columns)):
        ax = plt.subplot(6, 5, i)
        ax.title.set_text(categorial_columns[i])
        current_value_counts = df[categorial_columns[i]].value_counts()
        sns.barplot(x=current_value_counts.index, y=current_value_counts.values)

    plt.show()
    ''')

    fig = plt.figure(figsize=(20,25))
    for i in range(1, len(categorial_columns)):
        ax = plt.subplot(7, 3, i)
        ax.title.set_text(categorial_columns[i])
        current_value_counts = df[categorial_columns[i]].value_counts()
        sns.barplot(x=current_value_counts.index, y=current_value_counts.values)

    st.pyplot(fig)

st.markdown("# 4. Build PCA model [Returns to Education]")
st.markdown('[Google Colab](https://colab.research.google.com/drive/1uQ0imXLea3DdBIZOevHrnXEV9-TKawcp?usp=sharing)', unsafe_allow_html=True)