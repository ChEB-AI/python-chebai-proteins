from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd


class MockOntologyGraphData(ABC):
    """
    Abstract base class for mocking ontology graph data.

    This class provides a set of static methods that must be implemented by subclasses
    to return various elements of an ontology graph such as nodes, edges, and dataframes.
    """

    @staticmethod
    @abstractmethod
    def get_nodes() -> List[int]:
        """
        Get a list of node IDs in the ontology graph.

        Returns:
            List[int]: A list of node IDs.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_nodes() -> int:
        """
        Get the number of nodes in the ontology graph.

        Returns:
            int: The total number of nodes.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_edges() -> Set[Tuple[int, int]]:
        """
        Get the set of edges in the ontology graph.

        Returns:
            Set[Tuple[int, int]]: A set of tuples where each tuple represents an edge between two nodes.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_edges() -> int:
        """
        Get the number of edges in the ontology graph.

        Returns:
            int: The total number of edges.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_edges_of_transitive_closure_graph() -> Set[Tuple[int, int]]:
        """
        Get the set of edges in the transitive closure of the ontology graph.

        Returns:
            Set[Tuple[int, int]]: A set of tuples representing the transitive closure edges.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_transitive_edges() -> int:
        """
        Get the number of edges in the transitive closure of the ontology graph.

        Returns:
            int: The total number of transitive edges.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_obsolete_nodes_ids() -> Set[int]:
        """
        Get the set of obsolete node IDs in the ontology graph.

        Returns:
            Set[int]: A set of obsolete node IDs.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transitively_closed_graph() -> nx.DiGraph:
        """
        Get the transitive closure of the ontology graph.

        Returns:
            nx.DiGraph: A directed graph representing the transitive closure of the ontology graph.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_data_in_dataframe() -> pd.DataFrame:
        """
        Get the ontology data as a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing ontology data.
        """
        pass


class GOUniProtMockData(MockOntologyGraphData):
    """
    A mock ontology representing a simplified version of the Gene Ontology (GO) structure with nodes and edges
    representing GO terms and their relationships in a directed acyclic graph (DAG).

    Nodes:
        - GO_1
        - GO_2
        - GO_3
        - GO_4
        - GO_5
        - GO_6

    Edges (Parent-Child Relationships):
        - GO_1 -> GO_2
        - GO_1 -> GO_3
        - GO_2 -> GO_4
        - GO_2 -> GO_5
        - GO_3 -> GO_4
        - GO_4 -> GO_6

    This mock ontology structure is useful for testing methods related to GO hierarchy, graph extraction, and transitive
    closure operations.

    The class also includes methods to retrieve nodes, edges, and transitive closure of the graph.

    Visual Representation Graph with Valid Nodes and Edges:

                                GO_1
                               /    \
                             GO_2   GO_3
                            /  \    /
                         GO_5   GO_4
                                   \
                                   GO_6

    Valid Swiss Proteins with mapping to valid GO ids
    Swiss_Prot_1 -> GO_2, GO_3, GO_5
    Swiss_Prot_2 -> GO_2, GO_5
    """

    @staticmethod
    def get_nodes() -> List[int]:
        """
        Get a sorted list of node IDs.

        Returns:
            List[int]: A sorted list of node IDs in the ontology graph.
        """
        return sorted([1, 2, 3, 4, 5, 6])

    @staticmethod
    def get_number_of_nodes() -> int:
        """
        Get the total number of nodes in the ontology graph.

        Returns:
            int: The number of nodes.
        """
        return len(GOUniProtMockData.get_nodes())

    @staticmethod
    def get_edges() -> Set[Tuple[int, int]]:
        """
        Get the set of edges in the ontology graph.

        Returns:
            Set[Tuple[int, int]]: A set of tuples where each tuple represents an edge between two nodes.
        """
        return {(1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (4, 6)}

    @staticmethod
    def get_number_of_edges() -> int:
        """
        Get the total number of edges in the ontology graph.

        Returns:
            int: The number of edges.
        """
        return len(GOUniProtMockData.get_edges())

    @staticmethod
    def get_edges_of_transitive_closure_graph() -> Set[Tuple[int, int]]:
        """
        Get the set of edges in the transitive closure of the ontology graph.

        Returns:
            Set[Tuple[int, int]]: A set of tuples representing edges in the transitive closure graph.
        """
        return {
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 4),
            (3, 6),
            (4, 6),
        }

    @staticmethod
    def get_number_of_transitive_edges() -> int:
        """
        Get the total number of edges in the transitive closure graph.

        Returns:
            int: The number of transitive edges.
        """
        return len(GOUniProtMockData.get_edges_of_transitive_closure_graph())

    @staticmethod
    def get_obsolete_nodes_ids() -> Set[int]:
        """
        Get the set of obsolete node IDs in the ontology graph.

        Returns:
            Set[int]: A set of node IDs representing obsolete nodes.
        """
        return {7, 8}

    @staticmethod
    def get_GO_raw_data() -> str:
        """
        Get raw data in string format for a basic Gene Ontology (GO) structure.

        This data simulates a basic GO ontology format typically used for testing purposes.
        The data will include valid and obsolete GO terms with various relationships between them.

        Scenarios covered:
            - Obsolete terms being the parent of valid terms.
            - Valid terms being the parent of obsolete terms.
            - Both direct and indirect hierarchical relationships between terms.

        The data is designed to help test the proper handling of obsolete and valid GO terms,
        ensuring that the ontology parser can correctly manage both cases.

        Returns:
            str: The raw GO data in string format, structured as test input.
        """
        return """
        [Term]
        id: GO:0000001
        name: GO_1
        namespace: molecular_function
        def: "OBSOLETE. Assists in the correct assembly of ribosomes or ribosomal subunits in vivo, but is not a component of the assembled ribosome when performing its normal biological function." [GOC:jl, PMID:12150913]
        comment: This term was made obsolete because it refers to a class of gene products and a biological process rather than a molecular function.
        synonym: "ribosomal chaperone activity" EXACT []
        xref: MetaCyc:BETAGALACTOSID-RXN
        xref: Reactome:R-HSA-189062 "lactose + H2O => D-glucose + D-galactose"
        xref: Reactome:R-HSA-5658001 "Defective LCT does not hydrolyze Lac"
        xref: RHEA:10076

        [Term]
        id: GO:0000002
        name: GO_2
        namespace: biological_process
        is_a: GO:0000001 ! hydrolase activity, hydrolyzing O-glycosyl compounds
        is_a: GO:0000008 ! hydrolase activity, hydrolyzing O-glycosyl compounds

        [Term]
        id: GO:0000003
        name: GO_3
        namespace: cellular_component
        is_a: GO:0000001 ! regulation of DNA recombination

        [Term]
        id: GO:0000004
        name: GO_4
        namespace: biological_process
        is_a: GO:0000003 ! regulation of DNA recombination
        is_a: GO:0000002 ! hydrolase activity, hydrolyzing O-glycosyl compounds

        [Term]
        id: GO:0000005
        name: GO_5
        namespace: molecular_function
        is_a: GO:0000002 ! regulation of DNA recombination

        [Term]
        id: GO:0000006
        name: GO_6
        namespace: cellular_component
        is_a: GO:0000004 ! glucoside transport

        [Term]
        id: GO:0000007
        name: GO_7
        namespace: biological_process
        is_a: GO:0000003 ! glucoside transport
        is_obsolete: true

        [Term]
        id: GO:0000008
        name: GO_8
        namespace: molecular_function
        is_obsolete: true

        [Typedef]
        id: term_tracker_item
        name: term tracker item
        namespace: external
        xref: IAO:0000233
        is_metadata_tag: true
        is_class_level: true
        """

    @staticmethod
    def protein_sequences() -> Dict[str, str]:
        """
        Get the protein sequences for Swiss-Prot proteins.

        Returns:
            Dict[str, str]: A dictionary where keys are Swiss-Prot IDs and values are their respective sequences.
        """
        return {
            "Swiss_Prot_1": "MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK".replace(
                " ", ""
            ),
            "Swiss_Prot_2": "EKGLIVGHFS GIKYKGEKAQ ASEVDVNKMC CWVSKFKDAM RRYQGIQTCK".replace(
                " ", ""
            ),
        }

    @staticmethod
    def proteins_for_pretraining() -> List[str]:
        """
        Returns a list of protein IDs which will be used for pretraining based on mock UniProt data.

        Proteins include those with:
        - No GO classes or invalid GO classes (missing required evidence codes).

        Returns:
            List[str]: A list of protein IDs that do not meet validation criteria.
        """
        return [
            "Swiss_Prot_5",  # No GO classes associated
            "Swiss_Prot_6",  # GO class with no evidence code
            "Swiss_Prot_7",  # GO class with invalid evidence code
        ]

    @staticmethod
    def get_UniProt_raw_data() -> str:
        """
        Get raw data in string format for UniProt proteins.

        This mock data contains eleven Swiss-Prot proteins with different properties:
        - **Swiss_Prot_1**: A valid protein with three valid GO classes and one invalid GO class.
        - **Swiss_Prot_2**: Another valid protein with two valid GO classes and one invalid.
        - **Swiss_Prot_3**: Contains valid GO classes but has a sequence length > 1002.
        - **Swiss_Prot_4**: Has valid GO classes but contains an invalid amino acid, 'B'.
        - **Swiss_Prot_5**: Has a sequence but no GO classes associated.
        - **Swiss_Prot_6**: Has GO classes without any associated evidence codes.
        - **Swiss_Prot_7**: Has a GO class with an invalid evidence code.
        - **Swiss_Prot_8**: Has a sequence length > 1002 and has only invalid GO class.
        - **Swiss_Prot_9**: Has no GO classes but contains an invalid amino acid, 'B', in its sequence.
        - **Swiss_Prot_10**: Has a valid GO class but lacks a sequence.
        - **Swiss_Prot_11**: Has only Invalid GO class but lacks a sequence.

        Note:
        A valid GO label is the one which has one of the following evidence code specified in
        go_uniprot.py->`EXPERIMENTAL_EVIDENCE_CODES`.
        Invalid amino acids are specified in go_uniprot.py->`AMBIGUOUS_AMINO_ACIDS`.

        Returns:
            str: The raw UniProt data in string format.
        """
        protein_sq_1 = GOUniProtMockData.protein_sequences()["Swiss_Prot_1"]
        protein_sq_2 = GOUniProtMockData.protein_sequences()["Swiss_Prot_2"]
        raw_str = (
            # Below protein with 3 valid associated GO class and one invalid GO class
            f"ID   Swiss_Prot_1              Reviewed;         {len(protein_sq_1)} AA. \n"
            "AC   Q6GZX4;\n"
            "DR   GO; GO:0000002; C:membrane; EXP:UniProtKB-KW.\n"
            "DR   GO; GO:0000003; C:membrane; IDA:UniProtKB-KW.\n"
            "DR   GO; GO:0000005; P:regulation of viral transcription; IPI:InterPro.\n"
            "DR   GO; GO:0000004; P:regulation of viral transcription; IEA:SGD.\n"
            f"SQ   SEQUENCE   {len(protein_sq_1)} AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            f"     {protein_sq_1}\n"
            "//\n"
            # Below protein with 2 valid associated GO class and one invalid GO class
            f"ID   Swiss_Prot_2              Reviewed;         {len(protein_sq_2)} AA.\n"
            "AC   DCGZX4;\n"
            "DR   EMBL; AY548484; AAT09660.1; -; Genomic_DNA.\n"
            "DR   GO; GO:0000002; P:regulation of viral transcription; IMP:InterPro.\n"
            "DR   GO; GO:0000005; P:regulation of viral transcription; IGI:InterPro.\n"
            "DR   GO; GO:0000006; P:regulation of viral transcription; IEA:PomBase.\n"
            f"SQ   SEQUENCE   {len(protein_sq_2)} AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            f"     {protein_sq_2}\n"
            "//\n"
            # Below protein with all valid associated GO class but sequence length greater than 1002
            f"ID   Swiss_Prot_3              Reviewed;         {len(protein_sq_1 * 25)} AA.\n"
            "AC   Q6GZX4;\n"
            "DR   EMBL; AY548484; AAT09660.1; -; Genomic_DNA.\n"
            "DR   GO; GO:0000002; P:regulation of viral transcription; IEP:InterPro.\n"
            "DR   GO; GO:0000005; P:regulation of viral transcription; TAS:InterPro.\n"
            "DR   GO; GO:0000006; P:regulation of viral transcription; EXP:PomBase.\n"
            f"SQ   SEQUENCE   {len(protein_sq_1 * 25)} AA;  129118 MW;  FE2984658CED53A8 CRC64;\n"
            f"     {protein_sq_1 * 25}\n"
            "//\n"
            # Below protein has valid go class association but invalid amino acid `X` in its sequence
            "ID   Swiss_Prot_4              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "DR   EMBL; AY548484; AAT09660.1; -; Genomic_DNA.\n"
            "DR   GO; GO:0000002; P:regulation of viral transcription; EXP:InterPro.\n"
            "DR   GO; GO:0000005; P:regulation of viral transcription; IEA:InterPro.\n"
            "DR   GO; GO:0000006; P:regulation of viral transcription; EXP:PomBase.\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     BAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS\n"
            "//\n"
            # Below protein with sequence string but has no GO class
            "ID   Swiss_Prot_5              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "DR   EMBL; AY548484; AAT09660.1; -; Genomic_DNA.\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS\n"
            "//\n"
            # Below protein with sequence string and with NO `valid` associated GO class (no evidence code)
            "ID   Swiss_Prot_6              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "DR   GO; GO:0000023; P:regulation of viral transcription;\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS\n"
            "//\n"
            # Below protein with sequence string and with NO `valid` associated GO class (invalid evidence code)
            "ID   Swiss_Prot_7              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "DR   GO; GO:0000024; P:regulation of viral transcription; IEA:SGD.\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS\n"
            "//\n"
            # Below protein with sequence length greater than 1002 but with `Invalid` associated GO class
            f"ID   Swiss_Prot_8              Reviewed;         {len(protein_sq_2 * 25)} AA.\n"
            "AC   Q6GZX4;\n"
            "DR   GO; GO:0000025; P:regulation of viral transcription; IC:Inferred.\n"
            f"SQ   SEQUENCE   {len(protein_sq_2 * 25)} AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            f"     {protein_sq_2 * 25}\n"
            "//\n"
            # Below protein with sequence string but invalid amino acid `X` in its sequence
            "ID   Swiss_Prot_9              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     BAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS\n"
            "//\n"
            # Below protein with a `valid` associated GO class but without sequence string
            "ID   Swiss_Prot_10              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "DR   GO; GO:0000027; P:regulation of viral transcription; EXP:InterPro.\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     \n"
            "//\n"
            # Below protein with a `Invalid` associated GO class but without sequence string
            "ID   Swiss_Prot_11              Reviewed;         60 AA.\n"
            "AC   Q6GZX4;\n"
            "DR   GO; GO:0000028; P:regulation of viral transcription; ND:NoData.\n"
            "SQ   SEQUENCE   60 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
            "     \n"
            "//\n"
        )

        return raw_str

    @staticmethod
    def get_data_in_dataframe() -> pd.DataFrame:
        """
        Get a mock DataFrame representing UniProt data.

        The DataFrame contains Swiss-Prot protein data, including identifiers, accessions, GO terms, sequences,
        and binary label columns representing whether each protein is associated with certain GO classes.

        Returns:
            pd.DataFrame: A DataFrame containing mock UniProt data with columns for 'swiss_id', 'accession', 'go_ids', 'sequence',
                          and binary labels for GO classes.
        """
        expected_data = OrderedDict(
            swiss_id=["Swiss_Prot_1", "Swiss_Prot_2"],
            accession=["Q6GZX4", "DCGZX4"],
            go_ids=[[1, 2, 3, 5], [1, 2, 5]],
            sequence=list(GOUniProtMockData.protein_sequences().values()),
            **{
                #   SP_1,  SP_2
                1: [True, True],
                2: [True, True],
                3: [True, False],
                4: [False, False],
                5: [True, True],
                6: [False, False],
            },
        )
        return pd.DataFrame(expected_data)

    @staticmethod
    def get_transitively_closed_graph() -> nx.DiGraph:
        """
        Get the transitive closure of the ontology graph.

        Returns:
            nx.DiGraph: A directed graph representing the transitive closure of the ontology graph.
        """
        g = nx.DiGraph()
        g.add_nodes_from(node for node in ChebiMockOntology.get_nodes())
        g.add_edges_from(GOUniProtMockData.get_edges_of_transitive_closure_graph())
        return g
