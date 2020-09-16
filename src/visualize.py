import pydot
from IPython.display import SVG, display

def get_label(layer):
    nO = layer.get_dim("nO") if layer.has_dim("nO") else "?"
    nI = layer.get_dim("nI") if layer.has_dim("nI") else "?"
    return f"{layer.name}|({nI}, {nO})".replace(">", "&gt;")

def visualize_model(model):
    dot = pydot.Dot()
    dot.set("rankdir", "LR")
    dot.set_node_defaults(shape="record", fontname="arial", fontsize="10")
    dot.set_edge_defaults(arrowsize="0.7")
    nodes = {}
    for i, layer in enumerate(model.layers):
        label = get_label(layer)
        node = pydot.Node(layer.id, label=label)
        dot.add_node(node)
        nodes[layer.id] = node
        if i == 0:
            continue
        from_node = nodes[model.layers[i - 1].id]
        to_node = nodes[layer.id]
        if not dot.get_edge(from_node, to_node):
            dot.add_edge(pydot.Edge(from_node, to_node))
    display(SVG(dot.create_svg()))

