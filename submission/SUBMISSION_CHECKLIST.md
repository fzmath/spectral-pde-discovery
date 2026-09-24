# AMC (Applied Mathematics and Computation) 投稿检查清单

## 一、投稿系统文件清单

### 1. 必交文件
- [ ] **Manuscript PDF**（编译后的完整PDF，含图表、参考文献）
  - 用 `pdflatex main → bibtex main → pdflatex main → pdflatex main` 顺序编译
  - 检查无 undefined references、无 overfull hbox 警告（可接受少量）
  - 检查无空白页、无问号（编译残留）
- [ ] **LaTeX 源文件包**（main.tex + refs.bib + 图片）
  - AMC 接受 LaTeX 投稿，但首次投稿上传 PDF 即可，源文件修回时再传
- [ ] **Cover Letter**（cover_letter.pdf）
  - 已写好 cover_letter.tex，需编译成 PDF
  - 内容：投稿声明、三点创新、与期刊匹配、无一稿多投

### 2. 建议交文件
- [ ] **Highlights**（highlights.docx 或 txt）
  - 3-5 条，每条 ≤ 85 字符（含空格）
  - 已写好 highlights.md
- [ ] **Supplementary Material**（可选）
  - 源代码（code/ 文件夹打包成 zip）
  - 实验数据 JSON 文件
  - AMC 不强制要求，但有代码可提高可信度

---

## 二、作者信息与元数据

### 投稿系统填写
- [ ] **文章标题**：Spatiotemporal spectral derivatives with forward stepwise BIC for robust partial differential equation discovery
- [ ] **作者列表**：
  - Fuchang Wang¹,*（通讯作者）
  - Huirong Cao²
- [ ] **单位**：
  - ¹ School of Science, Emergency Management University, No. 465 College Road, Sanhe, Hebei 065201, China
  - ² School of Science, Langfang Normal University, No. 100 Ai-min Road, Langfang, Hebei 065000, China
- [ ] **通讯作者邮箱**：wangfuchang@cidp.edu.cn
- [ ] **摘要**：已精炼，约200词
- [ ] **关键词**（5-6个）：
  - PDE discovery
  - Spectral methods
  - Sparse regression
  - Bayesian information criterion
  - Non-periodic problems
- [ ] **MSC 分类号**（Mathematics Subject Classification）：
  - 65M70 (Spectral methods)
  - 65N35 (Spectral methods in PDE)
  - 68T07 (Machine learning)
  - 41A10 (Approximation by polynomials)

---

## 三、论文格式检查

### LaTeX 模板
- [ ] 使用 elsarticle 文档类（AMC 官方模板）
- [ ] `\journal{Applied Mathematics and Computation}` 已设置
- [ ] 字体大小 12pt，preprint 模式（首次投稿单栏即可）

### 图表
- [ ] 所有图片同时提供 PDF（矢量）和 EPS 格式
- [ ] 图片分辨率：PDF/EPS 矢量图无需 DPI 设置
- [ ] 图表标题在图片下方（`\caption`）
- [ ] 表格使用 `booktabs`（\toprule, \midrule, \bottomrule）
- [ ] 检查表格是否超出页面右边界（overfull hbox）
- [ ] 所有图表在正文中被引用（`\ref`）
- [ ] 图1-4 已更新（含 Spatial Spectral 基线）

### 公式
- [ ] 所有公式有编号（如需引用）
- [ ] 缩写首次出现给全称：
  - PDE = Partial Differential Equation
  - FFT = Fast Fourier Transform
  - BIC = Bayesian Information Criterion
  - STLSQ = Sequentially Thresholded Least Squares
  - FD = Finite Difference
  - SNR = Signal-to-Noise Ratio

### 参考文献
- [ ] 使用 BibTeX（refs.bib）
- [ ] 引用格式：Elsevier numeric style（[1], [2]...）
- [ ] 所有引用在正文中出现，无 undefined citation
- [ ] 所有参考文献真实存在（已核验17篇）
- [ ] arXiv 预印本已更新为正式期刊（如有）
- [ ]  DOI 链接格式统一

---

## 四、声明部分（Declarations）

### 必须在论文末尾包含
- [ ] **Data availability statement**：代码和数据在 GitHub 公开
- [ ] **Declaration of competing interest**：声明无利益冲突
- [ ] **CRediT authorship contribution statement**：
  - Fuchang Wang: Conceptualization, Methodology, Software, Writing - original draft
  - Huirong Cao: Validation, Writing - review & editing
- [ ] **Acknowledgment**：基金资助
  - 廊坊市科学技术研究与发展计划项目 2024011015

### Generative AI 声明
- [ ] 如果使用了 AI 辅助写作，需在参考文献前加声明
- [ ] AMC 要求："We used [AI tool] to [task] in order to [purpose]. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication."

---

## 五、推荐审稿人（可选但建议）

AMC 投稿系统可能要求建议 3-5 个审稿人：
- [ ] Prof. Steven L. Brunton (University of Washington, USA) — SINDy 奠基人
- [ ] Prof. J. Nathan Kutz (University of Washington, USA) — SINDy 合作者
- [ ] Dr. Daniel A. Messenger (Boston University, USA) — Weak SINDy 作者
- [ ] Dr. Samuel H. Schoenbaum (NIH, USA) — 或 Schaeffer 2017 作者
- [ ] 注意：不要建议同单位合作者、近期合作者、导师/学生

---

## 六、投稿前最终检查

- [ ] 通读全文，检查逻辑是否连贯
- [ ] 检查是否有 AI 痕迹（模板化表达、过度使用 "Furthermore/Moreover"、每段开头相同结构）
- [ ] 检查数字一致性：摘要中的数字与表格一致
- [ ] 检查符号一致性：测试函数用 ψ（不用 φ，避免与滤波器冲突）
- [ ] 检查 Limitations 诚实说明（单次运行、2D弱形式未实现等）
- [ ] 编译 PDF 后通读一遍，检查排版
- [ ] 确认 Cover Letter 已编译为 PDF
- [ ] 确认 Highlights 已准备

---

## 七、投稿后

- [ ] 记录 Manuscript ID
- [ ] 确认收到确认邮件
- [ ] 预计初审周期：2-4 周
- [ ] 预计总周期：6-12 个月
