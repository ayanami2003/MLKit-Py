from typing import Optional, Dict, Any, List
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import uuid
import json

class AutomatedLineageTracker(BaseValidator):

    def __init__(self, name: Optional[str]=None, track_derivatives: bool=True, capture_metadata: bool=True):
        """
        Initialize the AutomatedLineageTracker.
        
        Parameters
        ----------
        name : Optional[str]
            Name identifier for this tracker instance
        track_derivatives : bool
            Whether to track derived data products and their relationships
        capture_metadata : bool
            Whether to capture detailed metadata about transformations
        """
        super().__init__(name)
        self.lineage_graph: Dict[str, Any] = {'sources': {}, 'transformations': [], 'nodes': {}, 'edges': []}
        self.tracking_enabled = True
        self.track_derivatives = track_derivatives
        self.capture_metadata = capture_metadata
        self._data_sources: Dict[str, Dict[str, Any]] = {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that data lineage is being properly tracked.
        
        This method checks if the data batch contains proper lineage information
        and verifies the integrity of lineage metadata.
        
        Parameters
        ----------
        data : DataBatch
            Data batch to validate lineage information for
        **kwargs : dict
            Additional validation parameters
            
        Returns
        -------
        bool
            True if lineage tracking is valid, False otherwise
        """
        self.reset_validation_state()
        if not data.metadata:
            self.add_error('Data batch missing metadata')
            return False
        if 'lineage' not in data.metadata:
            self.add_error('Data batch missing lineage metadata')
            return False
        lineage_info = data.metadata['lineage']
        if not isinstance(lineage_info, list):
            self.add_error('Lineage info must be a list')
            return False
        if len(lineage_info) == 0:
            return True
        for trace_entry in lineage_info:
            if not isinstance(trace_entry, dict):
                self.add_error('Each trace entry must be a dictionary')
                return False
            required_fields = ['transformation_id', 'transformer', 'input_batch', 'output_batch']
            for field in required_fields:
                if field not in trace_entry:
                    self.add_error(f'Trace entry missing required field: {field}')
                    return False
                if trace_entry[field] is None or trace_entry[field] == '':
                    self.add_error(f'Trace entry field {field} cannot be None or empty')
                    return False
        return len(self.validation_errors) == 0

    def register_data_source(self, source_id: str, source_info: Dict[str, Any]) -> None:
        """
        Register a new data source in the lineage tracking system.
        
        Parameters
        ----------
        source_id : str
            Unique identifier for the data source
        source_info : Dict[str, Any]
            Information about the data source (type, location, schema, etc.)
        """
        if not self.tracking_enabled:
            return
        self._data_sources[source_id] = {'source_id': source_id, 'source_info': source_info, 'registration_time': str(uuid.uuid4())}
        self.lineage_graph['sources'][source_id] = source_info
        self.lineage_graph['nodes'][source_id] = {'type': 'data_source', 'source_info': source_info}

    def track_transformation(self, input_data: DataBatch, output_data: DataBatch, transformer_info: Dict[str, Any]) -> DataBatch:
        """
        Track a transformation operation between input and output data.
        
        This method records the relationship between input and output data batches,
        capturing information about what transformation was applied.
        
        Parameters
        ----------
        input_data : DataBatch
            Input data batch before transformation
        output_data : DataBatch
            Output data batch after transformation
        transformer_info : Dict[str, Any]
            Information about the transformation component (name, parameters, etc.)
            
        Returns
        -------
        DataBatch
            Output data batch with updated lineage metadata
        """
        if not self.tracking_enabled:
            return output_data
        if output_data.batch_id is None:
            output_data.batch_id = str(uuid.uuid4())
        if input_data.batch_id is None:
            input_data.batch_id = str(uuid.uuid4())
        transformation_id = str(uuid.uuid4())
        transformation_record = {'transformation_id': transformation_id, 'input_batch': input_data.batch_id, 'output_batch': output_data.batch_id, 'transformer': transformer_info, 'timestamp': str(uuid.uuid4())}
        if self.capture_metadata:
            input_shape = getattr(input_data, 'shape', None)
            if input_shape is None and hasattr(input_data, 'get_shape'):
                input_shape = input_data.get_shape()
            output_shape = getattr(output_data, 'shape', None)
            if output_shape is None and hasattr(output_data, 'get_shape'):
                output_shape = output_data.get_shape()
            transformation_record['metadata'] = {'input_shape': input_shape, 'output_shape': output_shape}
        self.lineage_graph['transformations'].append(transformation_record)
        self.lineage_graph['edges'].append({'source': input_data.batch_id, 'target': output_data.batch_id, 'transformation_id': transformation_id})
        if output_data.metadata is None:
            output_data.metadata = {}
        if input_data.metadata and 'lineage' in input_data.metadata:
            trace = input_data.metadata['lineage'].copy()
        else:
            trace = []
        trace_entry = {'transformation_id': transformation_id, 'transformer': transformer_info, 'input_batch': input_data.batch_id, 'output_batch': output_data.batch_id}
        trace.append(trace_entry)
        output_data.metadata['lineage'] = trace
        if input_data.batch_id not in self.lineage_graph['nodes']:
            shape = getattr(input_data, 'shape', None)
            if shape is None and hasattr(input_data, 'get_shape'):
                shape = input_data.get_shape()
            self.lineage_graph['nodes'][input_data.batch_id] = {'type': 'data_batch', 'shape': shape}
        if output_data.batch_id not in self.lineage_graph['nodes']:
            shape = getattr(output_data, 'shape', None)
            if shape is None and hasattr(output_data, 'get_shape'):
                shape = output_data.get_shape()
            self.lineage_graph['nodes'][output_data.batch_id] = {'type': 'data_batch', 'shape': shape}
        return output_data

    def get_lineage_trace(self, data_batch: DataBatch) -> List[Dict[str, Any]]:
        """
        Retrieve the complete lineage trace for a data batch.
        
        Parameters
        ----------
        data_batch : DataBatch
            Data batch to retrieve lineage information for
            
        Returns
        -------
        List[Dict[str, Any]]
            Chronological list of transformation events in the data's lineage
        """
        if not data_batch.metadata or 'lineage' not in data_batch.metadata:
            return []
        return data_batch.metadata['lineage']

    def export_lineage_graph(self, format: str='json') -> Dict[str, Any]:
        """
        Export the complete lineage graph in a specified format.
        
        Parameters
        ----------
        format : str
            Export format ('json', 'dict', or other supported formats)
            
        Returns
        -------
        Dict[str, Any]
            Lineage graph representation in the requested format
        """
        if format.lower() == 'json':
            return json.loads(json.dumps({'sources': self.lineage_graph['sources'], 'transformations': self.lineage_graph['transformations'], 'nodes': self.lineage_graph['nodes'], 'edges': self.lineage_graph['edges']}, default=str))
        elif format.lower() in ['dict', 'dictionary']:
            return {'sources': self.lineage_graph['sources'], 'transformations': self.lineage_graph['transformations'], 'nodes': self.lineage_graph['nodes'], 'edges': self.lineage_graph['edges']}
        else:
            return {'sources': self.lineage_graph['sources'], 'transformations': self.lineage_graph['transformations'], 'nodes': self.lineage_graph['nodes'], 'edges': self.lineage_graph['edges']}