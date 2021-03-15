print("=======model parameters 1=============")
total_parameters = 0
group_params = {"modelembedding":0, "modelem/":0, "modelsignalm":0, "dense":0}
print("======trainable_variables")
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    print("name: {}, shape: {}, len_shape: {}, params: {}".format(variable.name, shape, len(shape), variable_parameters))
    total_parameters += variable_parameters
    for gname in group_params.keys():
        if variable.name.startswith(gname):
            group_params[gname] += variable_parameters
print("group_params: {}".format(group_params))
print("total_params: {}".format(total_parameters))
print("=======model parameters 1=============\n")


print("=======model parameters 2=============")
total_parameters = 0
group_params = {"modelembedding":0, "modelem/":0, "modelsignalm":0, "dense":0}
print("======global_variables")
for variable in tf.global_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    print("name: {}, shape: {}, len_shape: {}, params: {}".format(variable.name, shape, len(shape), variable_parameters))
    total_parameters += variable_parameters
    for gname in group_params.keys():
        if variable.name.startswith(gname):
            group_params[gname] += variable_parameters
print("group_params: {}".format(group_params))
print("total_params: {}".format(total_parameters))
print("=======model parameters 2=============\n")


print("=======model parameters 2.1=============")
total_parameters = 0
print("======global_variables")
total_paramters = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()])
print("total_params: {}".format(total_paramters))
print("=======model parameters 2.1=============\n")
