import sbt._

// To make sure that we have compatible versions of Cloud Client Libraries,
// we use the versions specified in the Google Cloud Libraries Bill of Materials (BOM).
// The libraries in the BOM don't have dependency conflicts that would manifest as
// NoSuchMethodError or NoClassDefFoundError.
// See https://storage.googleapis.com/cloud-opensource-java-dashboard/com.google.cloud/libraries-bom/index.html

// We use project/match_bom.py to pull the dependencies from the BOM.
// You will now need to manually ensure that all deps listed are added here and use Bill OF Materials (BOM)
// version and lock all deps to the version defined in a particular version of the BOM.
// See guidance in project/match_bom.py

object GCPDepsBOM {
  val v26_2_0_deps: List[ModuleID] = List(
    "com.google.cloud"   % "google-cloud-storage" % "2.16.0",
    "com.google.android" % "annotations"          % "4.1.1.4",
    // Below is manualy set to "1.31.1" instead of "2.1.1" as specified by BOM due to some transitive issues - :'(
    "com.google.api-client"    % "google-api-client"       % "1.31.1",
    "com.google.guava"         % "guava"                   % "31.1-jre",
    "com.google.code.findbugs" % "jsr305"                  % "3.0.2",
    "com.google.errorprone"    % "error_prone_annotations" % "2.16",
    "com.google.guava"         % "failureaccess"           % "1.0.1",
    "com.google.guava"  % "listenablefuture"   % "9999.0-empty-to-avoid-conflict-with-guava",
    "com.google.j2objc" % "j2objc-annotations" % "1.3",
    "com.google.http-client"  % "google-http-client-apache-v2"    % "1.42.3",
    "com.google.http-client"  % "google-http-client"              % "1.42.3",
    "com.google.http-client"  % "google-http-client-gson"         % "1.42.3",
    "com.google.code.gson"    % "gson"                            % "2.10",
    "com.google.oauth-client" % "google-oauth-client"             % "1.34.1",
    "com.google.api.grpc"     % "gapic-google-cloud-storage-v2"   % "2.16.0-alpha",
    "com.google.api.grpc"     % "grpc-google-cloud-storage-v2"    % "2.16.0-alpha",
    "com.google.api.grpc"     % "grpc-google-iam-v1"              % "1.6.22",
    "com.google.api.grpc"     % "proto-google-cloud-storage-v2"   % "2.16.0-alpha",
    "com.google.api.grpc"     % "proto-google-common-protos"      % "2.11.0",
    "com.google.protobuf"     % "protobuf-java"                   % "3.21.10",
    "com.google.api.grpc"     % "proto-google-iam-v1"             % "1.6.22",
    "com.google.api"          % "api-common"                      % "2.2.2",
    "com.google.api"          % "gax-grpc"                        % "2.20.1",
    "com.google.api"          % "gax-httpjson"                    % "0.105.1",
    "com.google.api"          % "gax"                             % "2.20.1",
    "com.google.apis"         % "google-api-services-storage"     % "v1-rev20220705-2.0.0",
    "com.google.auth"         % "google-auth-library-credentials" % "1.13.0",
    "com.google.auth"         % "google-auth-library-oauth2-http" % "1.13.0",
    "com.google.auto.value"   % "auto-value-annotations"          % "1.10.1",
    "com.google.cloud"        % "google-cloud-core-grpc"          % "2.9.0",
    "com.google.cloud"        % "google-cloud-core-http"          % "2.9.0",
    "com.google.cloud"        % "google-cloud-core"               % "2.9.0",
    "com.google.http-client"  % "google-http-client-appengine"    % "1.42.3",
    "com.google.http-client"  % "google-http-client-jackson2"     % "1.42.3",
    "com.google.protobuf"     % "protobuf-java-util"              % "3.21.10",
    "com.google.re2j"         % "re2j"                            % "1.6",
    "com.google.api-client"   % "google-api-client-servlet"       % "2.1.1",
    "com.google.oauth-client" % "google-oauth-client-servlet"     % "1.34.1",
  )
}
