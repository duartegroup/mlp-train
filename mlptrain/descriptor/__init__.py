from mlptrain.descriptor.soap_descriptor import SoapDescriptor


soap_descriptor = SoapDescriptor()
soap_matrix = soap_descriptor.compute_representation
soap_kernel_vector = soap_descriptor.kernel_vector


__all__ = ['soap_matrix', 'soap_kernel_vector']
