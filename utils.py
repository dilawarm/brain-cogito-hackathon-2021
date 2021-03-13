def normalize(data):
    try:
        data -= min(data)
        data /= max(data)
        return data
    except:
        data -= data.min()
        data /= data.max()
        return data
