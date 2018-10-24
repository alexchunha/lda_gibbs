#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Tokenizes the documents using the given vocabulary
def tokenize(docs, vocab):
    counts = [{word: len(re.findall(word, doc, flags = re.I))
               for word in vocab}
              for doc in docs]
    
    return [sum([[word] * count for word, count in doc.items()],
                [])
            for doc in counts]

#-- FUNCTIONS FOR TURNING STRINGS INTO HTML --#

# Turns a string into an HTML span with the specified properties
def style_html(text, color = '', background = '', bold = False, underline = False):
    base = "<span style = 'background: {}; " \
                               "color: {}; " \
                         "font-weight: {}; " \
                     "text-decoration: {}; " \
                         "font-family: monospace; font-size: 10pt;'>{}</span>"
    args = (background, color, 'bold' if bold else '',
            'underline' if underline else '', text)
    return base.format(*args)

html_space = style_html(' ')

# Highlights all occurences of `word` in `text` with the specified `color`
def hlite_word(word, color, text = None):
    return re.sub(word,
                  lambda m: style_html(m.group(0),
                                       color = 'white',
                                       background = color,
                                       bold = True),
                  text if text else word,
                  flags = re.I)

# Colorizes a single document
def colorize_doc(doc, vocab, topics, colors):
    topic_iter = iter(topics)
    doc = style_html(doc)
    for word in vocab:
        doc = re.sub(word,
                     lambda m: style_html(m.group(0),
                                          color = 'white',
                                          background = colors[next(topic_iter)],
                                          bold = True),
                     doc,
                     flags = re.I)
    return doc

#-- PLOTTING FUNCTIONS --#

# Compute and plot the histogram/distribution from the time-series data given
def plot_wdist(wdist, color = 'k', transform = lambda x: x, label = None):
    counts = Counter(wdist)
    normalization = sum(counts.values())
    keys = sorted(counts.keys())
    vals = [transform(counts[key] / normalization) for key in keys]
    plt.plot(keys, vals, 'o-', color = color, label = label)

# Plot a time series of topic assignments. Each vertical slice corresponds to
# a "sample" or assignment of topics at a particular point in time. Each cell
# corresponds to a particular word in the corpus, with the color representing
# the topic assigned to that word by the sampling algorithm.
def plot_topic_trajectory(image, aspect = 1, frac = 1):
    plt.figure(figsize = (10, 10))
    plt.axis('off')
    plt.text(-int(4 * aspect * frac), 5., 'Doc 2', fontsize = 15,
             rotation = 'vertical', verticalalignment = 'center')
    plt.text(-int(4 * aspect * frac), 17, 'Doc 1', fontsize = 15,
             rotation = 'vertical', verticalalignment = 'center')
    for i in range(0, int(100 * aspect * frac) + 1, 10 * aspect):
        plt.text(i, 25.5, i * 100 // 1000 // aspect, fontsize = 12,
                 horizontalalignment = 'center')
    plt.text(int(50 * aspect * frac), 29, 'passes / 1000', fontsize = 15,
             horizontalalignment = 'center')
    # Plot the trajectory with the specific resolution (up to a user-specified
    # fraction of 10k)
    plt.imshow(np.flip(np.array(image[:int(10000 * frac):(100 // aspect)]).T, axis = 0),
               cmap = 'plasma', aspect = aspect * frac)
