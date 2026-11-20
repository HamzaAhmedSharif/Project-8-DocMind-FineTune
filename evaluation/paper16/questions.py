"""Provenance: copied from Project 6 DocMind Agent `evaluation/questions.py`
at commit 2a91d77f46dd2bab46c5784b28b9e97b5d584f87. Do not improve the question set here.

Evaluation set for DocMind Agent. Source paper: 'Attention Is All You Need'.
doc    - answerable from the paper.
absent - the paper does not contain the answer; the correct behaviour is to say so.

Vendored into Project 8 from Project 7 evaluation/questions.py on 2026-11-20.
Only the 16 PAPER16 ids (d01-d12, a01-a04) are kept; web questions (w01-w04) are excluded.
"""

QUESTIONS = [
    {
  "id": "d01",
  "type": "doc",
  "question": "What is self-attention, as defined in the Transformer paper?",
  "key_facts": [
    {
      "fact": "relates different positions of a single sequence",
      "answer_regex": "different\\s*positions",
      "paper_regex": "different\\s*positions\\s*of\\s*a\\s*single\\s*sequence"
    },
    {
      "fact": "to compute a representation of the sequence",
      "answer_regex": "represent",
      "paper_regex": "compute\\s*a\\s*representation\\s*of\\s*the\\s*sequence"
    }
  ]
},
    {
  "id": "d02",
  "type": "doc",
  "question": "Which attention function does the Transformer use, and how is it computed?",
  "key_facts": [
    {
      "fact": "scaled dot-product attention",
      "answer_regex": "scaled\\s*dot[-\\s]*product",
      "paper_regex": "scaled\\s*dot-?\\s*product\\s*attention"
    },
    {
      "fact": "softmax over the scaled dot products",
      "answer_regex": "softmax",
      "paper_regex": "softmax"
    },
    {
      "fact": "scaling by the square root of the key dimension",
      "answer_regex": "square\\s*root|sqrt|√",
      "paper_regex": "divide\\s*each\\s*by"
    }
  ]
},
    {
  "id": "d03",
  "type": "doc",
  "question": "Why are the dot products scaled by the square root of the key dimension d_k?",
  "key_facts": [
    {
      "fact": "dot products grow large when d_k is large",
      "answer_regex": "(grow|become|get)\\s*large|large\\s*(in\\s*)?magnitude|large\\s*values",
      "paper_regex": "grow\\s*large\\s*in\\s*magnitude"
    },
    {
      "fact": "which pushes softmax into regions with extremely small gradients",
      "answer_regex": "small\\s*gradients?|vanishing|saturat",
      "paper_regex": "extremely\\s*small\\s*gradients"
    }
  ]
},
    {
  "id": "d04",
  "type": "doc",
  "question": "How many identical layers do the encoder and the decoder each have in the base Transformer, and what are the two sub-layers of an encoder layer?",
  "key_facts": [
    {
      "fact": "N = 6 identical layers",
      "answer_regex": "\\b(6|six)\\b",
      "paper_regex": "n\\s*=\\s*6\\s*identical\\s*layers"
    },
    {
      "fact": "a position-wise fully connected feed-forward sub-layer",
      "answer_regex": "feed[-\\s]*forward",
      "paper_regex": "position-?\\s*wise\\s*fully\\s*connected\\s*feed-?\\s*forward"
    }
  ]
},
    {
  "id": "d05",
  "type": "doc",
  "question": "What are the model dimension and the number of attention heads used in the base Transformer?",
  "key_facts": [
    {
      "fact": "d_model = 512",
      "answer_regex": "512",
      "paper_regex": "d\\s*_?\\s*model\\s*=\\s*512"
    },
    {
      "fact": "h = 8 parallel attention heads",
      "answer_regex": "\\b(8|eight)\\b",
      "paper_regex": "h\\s*=\\s*8\\s*parallel\\s*attention"
    }
  ]
},
    {
  "id": "d06",
  "type": "doc",
  "question": "What is multi-head attention and why does the paper use it?",
  "key_facts": [
    {
      "fact": "attends to different representation subspaces",
      "answer_regex": "(different|multiple)\\s*(representation\\s*)?subspaces?",
      "paper_regex": "different\\s*representation\\s*subspaces"
    },
    {
      "fact": "linearly projects queries, keys and values",
      "answer_regex": "project",
      "paper_regex": "linearly\\s*project"
    }
  ]
},
    {
  "id": "d07",
  "type": "doc",
  "question": "How does the Transformer give the model information about the order of tokens?",
  "key_facts": [
    {
      "fact": "positional encodings",
      "answer_regex": "positional\\s*encoding",
      "paper_regex": "positional\\s*encodings"
    },
    {
      "fact": "sine and cosine functions of different frequencies",
      "answer_regex": "\\bsine?\\b|sinusoid",
      "paper_regex": "sine\\s*and\\s*cosine\\s*functions\\s*of\\s*different\\s*frequencies"
    },
    {
      "fact": "added to the input embeddings",
      "answer_regex": "\\badd(ed|s)?\\b|\\bsum(med)?\\b",
      "paper_regex": "to\\s*the\\s*input\\s*embeddings"
    }
  ]
},
    {
  "id": "d08",
  "type": "doc",
  "question": "Which optimizer and learning-rate warmup did the authors use to train the Transformer?",
  "key_facts": [
    {
      "fact": "Adam optimizer",
      "answer_regex": "adam",
      "paper_regex": "adam\\s*optimizer"
    },
    {
      "fact": "warmup_steps = 4000",
      "answer_regex": "4000|4,000",
      "paper_regex": "warmup[_\\s]*steps\\s*=\\s*4000"
    }
  ]
},
    {
  "id": "d09",
  "type": "doc",
  "question": "Which regularization techniques were used during training?",
  "key_facts": [
    {
      "fact": "residual dropout",
      "answer_regex": "dropout",
      "paper_regex": "residual\\s*dropout"
    },
    {
      "fact": "label smoothing",
      "answer_regex": "label\\s*smoothing",
      "paper_regex": "label\\s*smoothing\\s*of\\s*value"
    }
  ]
},
    {
  "id": "d10",
  "type": "doc",
  "question": "What BLEU scores did the big Transformer reach on WMT 2014 English-to-German and English-to-French?",
  "key_facts": [
    {
      "fact": "28.4 BLEU on English-to-German",
      "answer_regex": "28\\.4",
      "paper_regex": "28\\.4"
    },
    {
      "fact": "41.8 (or 41.0) BLEU on English-to-French",
      "answer_regex": "41\\.[08]",
      "paper_regex": "41\\.[08]"
    }
  ]
},
    {
  "id": "d11",
  "type": "doc",
  "question": "Why does the paper argue that self-attention is preferable to recurrent and convolutional layers?",
  "key_facts": [
    {
      "fact": "shorter path length between long-range dependencies",
      "answer_regex": "path|long-?\\s*range|dependenc",
      "paper_regex": "path\\s*length\\s*between\\s*long-?\\s*range\\s*dependencies"
    },
    {
      "fact": "more computation can be parallelized",
      "answer_regex": "parallel",
      "paper_regex": "computation\\s*that\\s*can\\s*be\\s*parallelized"
    }
  ]
},
    {
  "id": "d12",
  "type": "doc",
  "question": "What is the purpose of masking in the decoder's self-attention?",
  "key_facts": [
    {
      "fact": "positions cannot attend to subsequent positions",
      "answer_regex": "(subsequent|future|later|following)\\s*positions",
      "paper_regex": "subsequent\\s*positions"
    },
    {
      "fact": "predictions depend only on known outputs at earlier positions",
      "answer_regex": "less\\s*than|earlier|previous|preceding|known\\s*outputs",
      "paper_regex": "positions\\s*less\\s*than\\s*i"
    }
  ]
},
    {
  "id": "a01",
  "type": "absent",
  "absent_regex": [
    "imagenet",
    "image\\s*classification"
  ],
  "question": "According to the paper, what accuracy did the Transformer reach on the ImageNet image classification task?"
},
    {
  "id": "a02",
  "type": "absent",
  "absent_regex": [
    "chinese",
    "mandarin"
  ],
  "question": "According to the paper, what BLEU score did the Transformer reach on English-to-Chinese translation?"
},
    {
  "id": "a03",
  "type": "absent",
  "absent_regex": [
    "word\\s*error\\s*rate",
    "librispeech"
  ],
  "question": "According to the paper, what word error rate did the model reach on a speech recognition benchmark?"
},
    {
  "id": "a04",
  "type": "absent",
  "absent_regex": [
    "kwh",
    "kilowatt"
  ],
  "question": "According to the paper, how many kilowatt-hours of energy did training the big model consume?"
}
]
