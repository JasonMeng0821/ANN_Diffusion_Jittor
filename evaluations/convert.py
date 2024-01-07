import tensorflow as tf
import tf2onnx


INCEPTION_V3_PATH = "classify_image_graph_def.pb"
tf.compat.v1.disable_eager_execution()

with open(INCEPTION_V3_PATH, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.compat.v1.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph)
    model_proto = onnx_graph.make_model('test')
    with open('model.onnx', 'wb') as f:
        f.write(model_proto.SerializeToString())