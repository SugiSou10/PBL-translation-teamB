def remove_250():
    with open("../unigram-corpus/train.ja", "r") as ja, open("../unigram-corpus/train.en", "r") as en, \
    open("../unigram-corpus/clean250.ja", "w") as f1, open("../unigram-corpus/clean250.en", "w") as f2:
        for ja_line,en_line in zip(ja, en):
            ja_words = ja_line.strip().split(" ")
            en_words = en_line.strip().split(" ")
            if len(ja_words) < 250 and len(en_words) < 250:
                f1.write(ja_line)
                f2.write(en_line)

def main():
    remove_250()


if __name__ == '__main__':
    main()