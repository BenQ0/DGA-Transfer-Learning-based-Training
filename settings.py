USE_MULTIPROCESSING = True
EPOCHS = 100

valid_chars_binary = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
                      'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23,
                      'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34,
                      '8': 35, '9': 36, '-': 37, '_': 38}

valid_chars_mcc = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
                   'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23,
                   'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34,
                   '8': 35, '9': 36, '-': 37, '_': 38, '.': 39}

DYNDNS_TLDS = ['3utilities.com', 'bounceme.net', 'ddns.net', 'ddnsking.com', 'dnsalias.com', 'doesntexist.com',
               'duckdns.org', 'dynalias.com', 'dyndns.org', 'dynserv.com', 'dynu.net', 'freedynamicdns.net',
               'freedynamicdns.org', 'github.io', 'gotdns.ch', 'hopto.org', 'mooo.com', 'myddns.me', 'myftp.biz',
               'myftp.org', 'mynumber.org', 'myvnc.com', 'onthewifi.com', 'redirectme.net', 'servebeer.com',
               'serveblog.net', 'servecounterstrike.com', 'serveftp.com', 'servegame.com', 'servehalflife.com',
               'servehttp.com', 'serveirc.com', 'serveminecraft.net', 'servemp3.com', 'servepics.com', 'servequake.com',
               'sytes.net', 'viewdns.net', 'webhop.me', 'yi.org', 'zapto.org']

max_features_binary = len(valid_chars_binary) + 1  # +1 for padding token
max_features_mcc = len(valid_chars_mcc) + 1  # +1 for padding token
maxlen_binary = 63
maxlen_mcc = 253
class_weighting_power = 0.3

cls_token = 39  # token for [CLS] when using transformer based models
