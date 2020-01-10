from sys import argv

try:
    net_name = argv[1]
except:
    raise Exception("No net name given!")

try:
    hyperparameters_file_name = "nets/{}/hyperparameters.py".format(net_name)
    hyperparameters_file = open(hyperparameters_file_name, "r")
    hyperparameter_file_lines = hyperparameters_file.readlines()
    hyperparameters_file.close()
except:
    raise Exception("No net named \"{}\"!".format(net_name))

updates = argv[2:]

if len(updates) == 0:
    raise Exception("No updates given!")


new_file_contents = ""
hyperparameters, _, values = zip(*[(lambda update: update.rpartition("=")) \
    (update) for update in updates])
hyperparameters, values = list(hyperparameters), list(values)

for file_line in hyperparameter_file_lines:
    new_line = None

    if file_line.rpartition(" = ")[0] in hyperparameters:
        this_index = hyperparameters.index(file_line.rpartition(" = ")[0])
        new_line = "{} = {}\n".format(hyperparameters[this_index],
                                      values[this_index])
        hyperparameters.pop(this_index)
        values.pop(this_index)
    else:
        new_line = file_line
    new_file_contents += new_line

if len(hyperparameters) != 0:
    raise Exception("Invalid hyperparameter name: {}".format(hyperparameters[0]))

hyperparameters_file = open(hyperparameters_file_name, "w")
hyperparameters_file.write(new_file_contents)
hyperparameters_file.close()
