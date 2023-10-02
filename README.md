# VeloHARMO
## VeloHARMO is the pipeline for translation velocity harmonization from native organism to host organism. VeloHARMO consists of two crucial components: 1.Deep learning modelling for translation velocity prediction in both native and host organisms. 2.Genetic algorithm to generate the harmonized mRNA sequences from native organism to host organism.

<h1>Introduction</h1>

<p>Heterologous expression is a conventional strategy for expressing foreign genes encoding interested proteins in host organisms. However, achieving high protein production is challenging. Therefore, designing mRNA sequences to improve production efficiency while maintaining protein activity is crucial. In recent years, several algorithms, such as codon optimization and harmonization have been developed, but not always successful as codon usage frequency does not accurately reflect translation velocity. In this study, we proposed a novel concept called “translation velocity harmonization” and developed a pipeline: VeloHARMO for translation velocity harmonization, which involves fine-tuning protein co-translational folding into mRNA design and optimization for the first time. We firstly used Ribo-seq data to quantify translation velocity and analyzed protein structure features in E. coli and yeast from AlphFold. Subsequently, we constructed two deep learning model VeloGRU and VeloBERT, based on gate recurrent unit (GRU) and Bidirectional Encoder Representations from Transformers (BERT), capable of predicting translation velocity using amino acid,mRNA sequence, and protein structure features. Then, we implemented a genetic algorithm that allows the designed mRNA sequence to mimic the velocity pattern in E. coli when expressed in yeast (Translation velocity harmonization). In summary, our methodology represents a significant advancement in mRNA design by harmonizing translation velocity across diverse organisms to produce functional proteins.</p>

<h1>Software and Installation</h1>
<p><strong>Required Software:</strong></p>
<p>This pipeline was tested on ABCI supercomputer at AIST. For most of the tools, we installed via conda or bioconda, and the following software is required:</p>

  </li>
  <li>gffread
    <ul>
      <li>Install via bioconda: <code>conda install -c bioconda gffread</code></li>
    </ul>
  </li>

