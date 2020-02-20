# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:42:55 2020

@author: a6q
"""

#%%

import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('treebank')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')


input_text = input_text = 'A polymer is a large molecule, or macromolecule, composed of many repeated subunits. Due to their broad range of properties, both synthetic and natural polymers play essential and ubiquitous roles in everyday life. Polymers range from familiar synthetic plastics such as polystyrene to natural biopolymers such as DNA and proteins that are fundamental to biological structure and function. Polymers, both natural and synthetic, are created via polymerization of many small molecules, known as monomers. Their consequently large molecular mass, relative to small molecule compounds, produces unique physical properties including toughness, viscoelasticity, and a tendency to form glasses and semicrystalline structures rather than crystals. The terms polymer and resin are often synonymous with plastic.'
#"WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
#Polymers may be synthetic or natural. They exhibit a wide range of optoelectronic properties. Eric likes polymers. But he liikes oxides better. Oxides are very important. They are grown at CNMS."
tokens = nltk.word_tokenize(input_text)

tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)

from nltk.corpus import treebank
#t = treebank.parsed_sents('wsj_0001.mrg')[0]
t = entities
t.draw()