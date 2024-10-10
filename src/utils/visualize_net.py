from torchview import draw_graph
import os

def visualize_graphviz(model, 
                       input_data, 
                    #    input_size, 
                       png_name, 
                       device):
    print('visualizing the model using graphviz...')
    print(len(input_data))
    # print(input_size)
    model_graph = draw_graph(model, 
                             input_data=input_data,
                             save_graph=True,
                             filename=png_name,
                            #  input_size=input_size,
                             device=device)
    model_graph.visual_graph
    # model_graph.save(path_to_png)  # Save as a PNG image
