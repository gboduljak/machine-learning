from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

def getTreeAsImage(tree):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return (dot_data, Image(graph))