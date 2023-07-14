class_names_combined = ['User unknown', 'TemporaryUserProblem', 'Mailbox full',
                        'Spam', 'Policy', 'TransportProblem', 'Unclassified', 'Greylisted']
class_map_combined = {k: v for k, v in zip(class_names_combined, range(len(class_names_combined)))}

def num_combined_classes():
    return len(class_map_combined)

def encode_combined_label(label):
    if label == 'User unknown':
        return class_map_combined['User unknown']
    elif label == 'TemporaryUserProblem':
        return class_map_combined['TemporaryUserProblem'] 
    elif label == 'Mailbox full':
        return class_map_combined['Mailbox full']
    elif label == 'Spam':# or label == 'Greylisted':
        return class_map_combined['Spam'] 
    elif label == 'Greylisted':# or label == 'Greylisted':
        return class_map_combined['Greylisted'] 
    elif label == 'Policy':
        return class_map_combined['Policy'] 
    elif label == 'NoTransport' or label == 'NoRelay' or label == 'ServerTempError':
        return class_map_combined['TransportProblem'] 
    elif label == 'Unclassified' or label == 'OtherTemporaryError' or label == 'OtherPermanentError':
        return class_map_combined['Unclassified']
    else:
        raise ValueError('Unknown label! ' + label)

def decode_combined_label(c):
    return class_names_combined[c]

def combined_target_names():
    return [decode_combined_label(x) for x in range(num_combined_classes())]


class_names_simplified = ['UserUnknownEx', 'SpamEx', 'Mailbox full', 'Policy', 'TransportProblem', 'Unclassified']
class_map_simplified = {k: v for k, v in zip(class_names_simplified, range(len(class_names_simplified)))}

def num_simplified_classes():
    return len(class_map_simplified)

def encode_simplified_label(label):
    if label == 'User unknown' or label == 'TemporaryUserProblem':
        return class_map_simplified['UserUnknownEx']
    elif label == 'Mailbox full':
        return class_map_simplified['Mailbox full']
    elif label == 'Spam' or label == 'Greylisted':
        return class_map_simplified['SpamEx']
    elif label == 'Policy':
        return class_map_simplified['Policy'] 
    elif label == 'NoTransport' or label == 'NoRelay' or label == 'ServerTempError':
        return class_map_simplified['TransportProblem'] 
    elif label == 'Unclassified' or label == 'OtherTemporaryError' or label == 'OtherPermanentError':
        return class_map_simplified['Unclassified']
    else:
        raise ValueError('Unknown label! ' + label)

def decode_simplified_label(c):
    return class_names_simplified[c]

def simplified_target_names():
    return [decode_simplified_label(x) for x in range(num_simplified_classes())]