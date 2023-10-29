import argparse
from src.confluence import ConflunenceParser



if __name__ == "__main__":

    confluence_parser = ConflunenceParser()
    parser = argparse.ArgumentParser(description="""
                                     Example:
                                     python confluence.py -s SD -t 'Analysis Cycle Configuration Guidelines'
                                     python confluence.py -l 'configuration_guideline'
                                     """)

    parser.add_argument("-s", "--space", help="Document space. example: SD", required=False)
    parser.add_argument("-t", "--title", help="Document title, example: Analysis Cycle Configuration Guidelines", required=False)
    parser.add_argument("-l", "--label", help="Download all documents with given label, example: configuration_guideline", required=False)

    args =  parser.parse_args()
    if args.label:
        print("label is given, use it to download docs")
        confluence_parser.save_pages_by_label_pdf(args.label)
    elif args.space and args.title:
        confluence_parser.save_page_md(args.space, args.title)
    else:
        print(parser.print_help())
    