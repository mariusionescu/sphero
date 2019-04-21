from cortex import Cortex


processor = Cortex(None, None, n_children=1)
processor.save('cortex.data')

# processor = Cortex.load('cortex.data')

print(processor)

