"""
This file contains methods to transform the SQuAD dataset into a format that can be sent to the Pega KnowledgeBuddy API.
"""
import os
import json

def squad_to_knowledge(
        data : list, 
        source_json : str, 
        output_dir : str, 
        datasetname : str,
        destination: str,
        per_doc: bool=False,
        single_doc_source: bool=False,
        **kwargs):
    """
    Takes SQuAD dataset and then writes the document(s) to the specified location in a format that 
    is understood by the Pega API.
    
    Data:           The dev or train dataset downloaded from the SQuAD website.
    Source_json:    Path to a JSON file to be sent to the API, where all parameters except for content are already defined.
                    Since the file will be copied, you will not lose this file.
    Output_dir:     Output directory.
    Datasetname:    Either "dev" or "train"
    per_doc:        Whether to write the full dataset to one request, or a request for each document.
    single_doc_source (bool): If true, creates a separate datasource for each document.


    Returns: List of (text only) documents.
    """
    # assert datasetname in ["dev", "train"], "unknown split, use 'dev' or 'train'."

    # Retrieve the document by concatenating all paragraphs per document.
    documents = []
    for i, doc in enumerate(data):
        context = ""
        pgs = doc['paragraphs']
        for j, pg in enumerate(pgs):
            pg = pg['context']
            if j == 0:
                context = pg
            else:
                context = "\n".join([context,pg])
        documents.append(context)

    # Save the documents to JSON files.
    output_dir = destination
    
    # This can be put into a single API ingestion
    if per_doc==True:
        for i in range(len(documents)):
            empty_ingestion = source_json
            destination = os.path.join(output_dir,f"SQuAD_{datasetname}_{i}.json")
            data = json.load(open(empty_ingestion, "r"))
            data["text"][0]["content"] = documents[i]
            data["dataSource"] = f"SQuAD_{datasetname}"
            data["title"] = f"SQuAD_{datasetname}"
            if single_doc_source:
                data["dataSource"] = f"SQuAD_{datasetname}_{i}"
                data["title"] = f"SQuAD_{datasetname}_{i}"
            id_i = str(i)
            while len(id_i) < 3:
                id_i = f"0{id_i}"
            data["objectId"] = f"{datasetname}_doc_{id_i}"
            with open(destination, 'w', encoding='utf8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

    else:
        print("Saving all contexts to one request...")
        empty_ingestion = source_json
        destination = os.path.join(output_dir, f"SQuAD_{datasetname}_full.json")
        data=json.load(open(empty_ingestion, "r"))
        data["text"][0]["content"] = "\n".join(documents)
        data["dataSource"]= f"SQuAD_{datasetname}"
        data["title"] = f"SQuAD_{datasetname}"
        data["objectId"] = f"{datasetname}Full"
        with open(destination, "w", encoding='utf8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    
    print("Done")
    # Return documents
    return documents
        


