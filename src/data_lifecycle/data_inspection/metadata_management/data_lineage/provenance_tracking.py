from typing import Dict, Any, Optional, List
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import json
import uuid
from datetime import datetime

class ProvenanceTracker(BaseValidator):
    """
    Tracks and validates data provenance throughout the data lifecycle.
    
    This class maintains a record of data transformations and sources to ensure
    traceability and compliance. It extends BaseValidator to integrate seamlessly
    with the validation framework while providing specialized provenance tracking.
    
    Attributes:
        name (str): Name of the validator instance
        track_derivatives (bool): Whether to track derivative data products
        capture_metadata (bool): Whether to capture detailed metadata
        provenance_records (List[Dict[str, Any]]): Collection of provenance records
    """

    def __init__(self, name: Optional[str]=None, track_derivatives: bool=True, capture_metadata: bool=True):
        """
        Initialize the ProvenanceTracker.
        
        Args:
            name (Optional[str]): Name for this validator instance
            track_derivatives (bool): Whether to track derived data products
            capture_metadata (bool): Whether to capture detailed transformation metadata
        """
        super().__init__(name)
        self.track_derivatives = track_derivatives
        self.capture_metadata = capture_metadata
        self.provenance_records: List[Dict[str, Any]] = []

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that data has proper provenance tracking.
        
        This method checks if the provided data batch contains the necessary
        provenance information according to the tracker's configuration.
        
        Args:
            data (DataBatch): Data batch to validate
            **kwargs: Additional validation parameters
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        self.reset_validation_state()
        if not self.track_derivatives and (not self.capture_metadata):
            return True
        if data.metadata is None:
            if self.track_derivatives or self.capture_metadata:
                self.add_error('Data batch has no metadata')
                return False
            return True
        if self.track_derivatives:
            if 'provenance_id' not in data.metadata:
                self.add_error('Data batch missing provenance_id in metadata')
                return False
            if 'transformation_history' not in data.metadata:
                self.add_warning('Tracking derivatives enabled but no transformation history found')
        if self.capture_metadata:
            if 'detailed_provenance' not in data.metadata:
                self.add_warning('Metadata capture enabled but no detailed provenance found')
        return len(self.validation_errors) == 0

    def record_data_source(self, source_id: str, source_info: Dict[str, Any]) -> None:
        """
        Record a new data source in the provenance tracking system.
        
        Args:
            source_id (str): Unique identifier for the data source
            source_info (Dict[str, Any]): Information about the data source
        """
        record = {'event_type': 'data_source', 'source_id': source_id, 'timestamp': datetime.utcnow().isoformat(), 'source_info': source_info.copy() if source_info else {}}
        self.provenance_records.append(record)

    def track_transformation(self, input_data: DataBatch, output_data: DataBatch, transformer_info: Dict[str, Any]) -> DataBatch:
        """
        Track a transformation operation between input and output data.
        
        Records the transformation details and updates the output data batch
        with provenance metadata.
        
        Args:
            input_data (DataBatch): Input data before transformation
            output_data (DataBatch): Output data after transformation
            transformer_info (Dict[str, Any]): Information about the transformation
            
        Returns:
            DataBatch: Output data with updated provenance metadata
        """
        provenance_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        record = {'event_type': 'transformation', 'provenance_id': provenance_id, 'trace_id': trace_id, 'timestamp': datetime.utcnow().isoformat(), 'input_batch_id': input_data.batch_id, 'output_batch_id': output_data.batch_id, 'transformer_info': transformer_info.copy() if transformer_info else {}}
        if input_data.metadata and 'provenance_id' in input_data.metadata:
            record['input_provenance_id'] = input_data.metadata['provenance_id']
        if self.capture_metadata and input_data.metadata:
            record['input_metadata'] = input_data.metadata.copy()
        self.provenance_records.append(record)
        if output_data.metadata is None:
            output_data.metadata = {}
        output_data.metadata['provenance_id'] = provenance_id
        output_data.metadata['trace_id'] = trace_id
        output_data.metadata['transformation_timestamp'] = record['timestamp']
        output_data.metadata['provenance'] = {'trace_id': trace_id}
        if 'transformation_history' not in output_data.metadata:
            output_data.metadata['transformation_history'] = []
        history_entry = {'provenance_id': provenance_id, 'transformer_info': transformer_info.copy() if transformer_info else {}, 'timestamp': record['timestamp']}
        if input_data.batch_id:
            history_entry['input_batch_id'] = input_data.batch_id
        output_data.metadata['transformation_history'].append(history_entry)
        if self.capture_metadata:
            output_data.metadata['detailed_provenance'] = record
        return output_data

    def get_provenance_trace(self, data_batch: DataBatch) -> List[Dict[str, Any]]:
        """
        Retrieve the complete provenance trace for a data batch.
        
        Args:
            data_batch (DataBatch): Data batch to trace
            
        Returns:
            List[Dict[str, Any]]: Chronological list of provenance records
        """
        if not data_batch.metadata or 'provenance_id' not in data_batch.metadata:
            return []
        provenance_id = data_batch.metadata['provenance_id']
        trace = []
        for record in self.provenance_records:
            if record.get('provenance_id') == provenance_id:
                trace.append(record)
            elif record.get('input_provenance_id') == provenance_id:
                trace.append(record)
        trace.sort(key=lambda x: x.get('timestamp', ''))
        return trace

    def export_provenance_graph(self, format: str='json') -> Dict[str, Any]:
        """
        Export the complete provenance graph in the specified format.
        
        Args:
            format (str): Export format ('json', 'dict', etc.)
            
        Returns:
            Dict[str, Any]: Provenance graph representation
        """
        if format.lower() not in ['json', 'dict']:
            raise ValueError(f"Unsupported export format: {format}. Supported formats: 'json', 'dict'")
        graph = {'graph_metadata': {'export_timestamp': datetime.utcnow().isoformat(), 'tracker_name': self.name, 'total_records': len(self.provenance_records), 'tracking_derivatives': self.track_derivatives, 'capturing_metadata': self.capture_metadata}, 'nodes': [], 'edges': [], 'records': self.provenance_records.copy()}
        nodes = {}
        edges = []
        for record in self.provenance_records:
            event_type = record.get('event_type', 'unknown')
            if event_type == 'data_source':
                source_id = record.get('source_id')
                if source_id:
                    nodes[source_id] = {'id': source_id, 'type': 'data_source', 'label': f'Source: {source_id}', 'properties': record.get('source_info', {})}
            elif event_type == 'transformation':
                prov_id = record.get('provenance_id')
                input_prov_id = record.get('input_provenance_id')
                if prov_id:
                    nodes[prov_id] = {'id': prov_id, 'type': 'data_batch', 'label': f'Batch: {prov_id[:8]}...', 'properties': {'timestamp': record.get('timestamp'), 'batch_id': record.get('output_batch_id')}}
                    if input_prov_id:
                        edges.append({'source': input_prov_id, 'target': prov_id, 'type': 'transformation', 'properties': record.get('transformer_info', {})})
        graph['nodes'] = list(nodes.values())
        graph['edges'] = edges
        return graph