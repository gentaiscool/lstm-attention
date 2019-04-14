from keras.models import Model
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import pandas as pd
import math as math

intermediate_layer_model2 = Model(inputs=model.input,
                                 outputs=model.layers[2].output)

intermediate_layer_model1 = Model(inputs=model.input,
                                 outputs=model.layers[1].output)

def show(sentences): 
    fig, axn = plt.subplots(len(sentences), 1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=5, hspace=10)
    fig.tight_layout()

    for i, ax in enumerate(axn.flat):
        seq = sentences[i]
        words = seq.split(" ")
        arr = numpy.zeros(35)
        for j in range(len(words)):
            if words[j] in word_to_idx:
                arr[j] = word_to_idx[words[j].lower()]
            else:
                arr[j] = word_to_idx[""]

        arr = numpy.reshape(arr, (1, arr.shape[0]))
        intermediate_output2 = intermediate_layer_model2.predict(arr, verbose=0)
        intermediate_output1 = intermediate_layer_model1.predict(arr, verbose=0)
        print(seq, model.predict(arr))
        
        weights = intermediate_output2 / intermediate_output1
        val = []
        total = 0
        for j in range(len(words)):
            val.append(weights[0][j][0])
            total += weights[0][j][0]

        d = {}
        print(val)
        d[""] = pd.Series(val, index=words)

        df = pd.DataFrame(d)
        df.reindex(sentences[i].split(" "))
        df = df.transpose()

    #     sns.heatmap(df, ax=ax, annot=False, linewidths=.0, cbar_ax=cbar_ax if i else cbar_ax, cmap="RdBu_r",#"YlGnBu",
    #                   cbar_kws={"orientation": "vertical"})
        sns.heatmap(df, ax=ax, annot=False, cbar_ax=None, cbar=False, linewidths=.0, cmap="YlGnBu")#"Oranges")#"RdBu_r")
    
    return fig

stressed = ["Number one being employment after graduation",
        "My parents give me a lot of pressure",
        "So yeah this course is actually very difficult",
        "And I will be very stressed out",
        "uh Currently I'm very anxious about several things",
        "i am not stressed"]

unstressed = ["It is got a rate of amazing food the culture is very good",
        "Eh it makes me feel relaxed and uh I enjoy those things",
        "And it has a really calming effect",
        "It's really fun yeah",
        "I just played around a lot"]

figure = None
figure = show(stressed)
figure.savefig('graphs/stressed.pdf', format='pdf', dpi=300)
figure = show(unstressed)
figure.savefig('graphs/unstressed.pdf', format='pdf', dpi=300)

# figure = show(x)
# figure.savefig('graphs/x.pdf', format='pdf', dpi=300)