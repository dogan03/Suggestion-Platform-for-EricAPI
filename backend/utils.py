def getCount(freq_dict, subject, start_year, end_year):
    if subject not in freq_dict:
        return 0
    
    subject_freq = freq_dict[subject]
    
    if end_year not in subject_freq:
        return 0
    if start_year not in subject_freq:
        start_year = list(subject_freq.keys())[0]
    return subject_freq[end_year] - subject_freq[start_year]

def getAllCount(subject_list, freq_dict, start_year, end_year):
    all_count = 0
    for subject in subject_list:
        all_count += getCount(freq_dict, subject, start_year, end_year)
    return all_count