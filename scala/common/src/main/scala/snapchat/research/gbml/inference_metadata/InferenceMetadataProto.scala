// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.inference_metadata

object InferenceMetadataProto extends _root_.scalapb.GeneratedFileObject {
  lazy val dependencies: Seq[_root_.scalapb.GeneratedFileObject] = Seq.empty
  lazy val messagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      snapchat.research.gbml.inference_metadata.InferenceMetadata,
      snapchat.research.gbml.inference_metadata.InferenceOutput
    )
  private lazy val ProtoBytes: _root_.scala.Array[Byte] =
      scalapb.Encoding.fromBase64(scala.collection.immutable.Seq(
  """Ci9zbmFwY2hhdC9yZXNlYXJjaC9nYm1sL2luZmVyZW5jZV9tZXRhZGF0YS5wcm90bxIWc25hcGNoYXQucmVzZWFyY2guZ2Jtb
  CL4AgoRSW5mZXJlbmNlTWV0YWRhdGESzAEKJ25vZGVfdHlwZV90b19pbmZlcmVuY2VyX291dHB1dF9pbmZvX21hcBgBIAMoCzJQL
  nNuYXBjaGF0LnJlc2VhcmNoLmdibWwuSW5mZXJlbmNlTWV0YWRhdGEuTm9kZVR5cGVUb0luZmVyZW5jZXJPdXRwdXRJbmZvTWFwR
  W50cnlCJuI/IxIhbm9kZVR5cGVUb0luZmVyZW5jZXJPdXRwdXRJbmZvTWFwUiFub2RlVHlwZVRvSW5mZXJlbmNlck91dHB1dEluZ
  m9NYXAakwEKJk5vZGVUeXBlVG9JbmZlcmVuY2VyT3V0cHV0SW5mb01hcEVudHJ5EhoKA2tleRgBIAEoCUII4j8FEgNrZXlSA2tle
  RJJCgV2YWx1ZRgCIAEoCzInLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuSW5mZXJlbmNlT3V0cHV0QgriPwcSBXZhbHVlUgV2YWx1Z
  ToCOAEiwwEKD0luZmVyZW5jZU91dHB1dBJBCg9lbWJlZGRpbmdzX3BhdGgYASABKAlCE+I/EBIOZW1iZWRkaW5nc1BhdGhIAFIOZ
  W1iZWRkaW5nc1BhdGiIAQESRAoQcHJlZGljdGlvbnNfcGF0aBgCIAEoCUIU4j8REg9wcmVkaWN0aW9uc1BhdGhIAVIPcHJlZGljd
  GlvbnNQYXRoiAEBQhIKEF9lbWJlZGRpbmdzX3BhdGhCEwoRX3ByZWRpY3Rpb25zX3BhdGhiBnByb3RvMw=="""
      ).mkString)
  lazy val scalaDescriptor: _root_.scalapb.descriptors.FileDescriptor = {
    val scalaProto = com.google.protobuf.descriptor.FileDescriptorProto.parseFrom(ProtoBytes)
    _root_.scalapb.descriptors.FileDescriptor.buildFrom(scalaProto, dependencies.map(_.scalaDescriptor))
  }
  lazy val javaDescriptor: com.google.protobuf.Descriptors.FileDescriptor = {
    val javaProto = com.google.protobuf.DescriptorProtos.FileDescriptorProto.parseFrom(ProtoBytes)
    com.google.protobuf.Descriptors.FileDescriptor.buildFrom(javaProto, _root_.scala.Array(
    ))
  }
  @deprecated("Use javaDescriptor instead. In a future version this will refer to scalaDescriptor.", "ScalaPB 0.5.47")
  def descriptor: com.google.protobuf.Descriptors.FileDescriptor = javaDescriptor
}