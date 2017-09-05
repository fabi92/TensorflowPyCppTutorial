#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/features2d.hpp>


#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

#define SQR(x) ((x)*(x))

using namespace Eigen;
using namespace std;
using namespace cv;

tensorflow::Session* session;

vector<Mat> regressFeaturesTensorFlow(vector<Mat> &samples, bool showRecons)
{
    if (samples.empty()) throw runtime_error("Empty samples!");

    // Figure out dimensions
    size_t batch_size = 1000;

    int width = samples[0].size().width;
    int height = samples[0].size().height;
    int depth = samples[0].channels();

    int bytes_per_sample_in = width * height * depth * sizeof(float);
    int bytes_per_sample_out = 128 * sizeof(float);


    // Build 4-dim Tensor for input and get Eigen mapping
    tensorflow::TensorShape shape({batch_size, height, width, depth});
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, shape);
    vector<pair<string, tensorflow::Tensor> > input({ {"inputs",tensor} });
    auto mapping_in = tensor.tensor<float, 4>();

    // Define input/output tensors
    vector< string > labels( {"latent"} );
    if (showRecons) labels.push_back("reconstruction");
    vector<tensorflow::Tensor> cae_out;

    vector<Mat> feats;

    size_t nr_batches = samples.size() / batch_size + 1;
    for (size_t batch=0; batch < nr_batches; ++batch)
    {

        size_t base = batch*batch_size;
        if (base > samples.size()) continue;

        int samplesToProcess = std::min(samples.size()-base, batch_size);
        for (int s = 0; s < samplesToProcess; s++){
            memcpy(&mapping_in(s,0,0,0), samples[base+s].data, bytes_per_sample_in);
        }

        tensorflow::Status status = session->Run(input, labels, {}, &cae_out);
        if(!status.ok()) throw runtime_error("Could not run graph: " + status.error_message());


        auto mapping_out = cae_out[0].tensor<float,2>();   // Get 2-dim Eigen tensor mapping for output

        for (int s = 0; s < samplesToProcess; s++){
            Mat m(1, 128, CV_32F);
            memcpy(m.data, &mapping_out(s,0), bytes_per_sample_out);
            feats.push_back(m);
        }

        if (showRecons)
        {
            auto mapping_recon = cae_out[1].tensor<float,4>();   // Get 2-dim Eigen tensor mapping for output
            for (int s = 0; s < samplesToProcess; s++)
            {
                Mat r(64,64, CV_32FC(depth), &mapping_recon(s,0,0,0));
                memcpy(r.data, &mapping_recon(s,0,0,0), bytes_per_sample_in);

                Mat &i = samples[s];
                cv::imshow("inputs", i);
                cv::imshow("reconstruction", r);
                cv::waitKey();
            }
        }

    }

    return feats;

}

int main(int argc, char *argv[])
{
    const string network = "/home/fabi/GIT/TensorflowPyCppTutorial/cpp/models/cae.pb";

    tensorflow::SessionOptions opts;
    tensorflow::GPUOptions *gpu_opts = new tensorflow::GPUOptions();

    opts.config.set_allow_soft_placement(false); // Allow mixing CPU and GPU data
    gpu_opts->set_allow_growth(true);            // Don't hog the full GPU at once
    opts.config.set_allocated_gpu_options(gpu_opts);

    tensorflow::Status status = tensorflow::NewSession(opts, &session);
    if(!status.ok()) cerr << "Could not creat new sessions: " << status.error_message() << endl;

    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), network, &graph_def);
    if(!status.ok()) cerr << "Could not read Proto File: " << status.error_message() << endl;

    status = session->Create(graph_def);
    if(!status.ok()) cerr << "Could not create graph " << status.error_message() << endl;


    string rgb_1 = "/home/fabi/Datasets/SIXD/LINEMOD/test/scene_01/rgb/0000.png";
    Mat m_col_1 = imread(rgb_1);


    string rgb_2 = "/home/fabi/Datasets/SIXD/LINEMOD/test/scene_01/rgb/0001.png";
    Mat m_col_2 = imread(rgb_2);

    m_col_1.convertTo(m_col_1, CV_32FC3, 1.f/255.f);
    m_col_2.convertTo(m_col_2, CV_32FC3, 1.f/255.f);


    cv::resize(m_col_1, m_col_1, cv::Size(64,64),0,0);
    cv::resize(m_col_2, m_col_2, cv::Size(64,64),0,0);

    vector< Mat> samples({m_col_1, m_col_2});

    cout << "loaded. running regression of features..." << endl;
    vector<Mat> feats = regressFeaturesTensorFlow(samples, true);

    cout << "printing latents" << endl;
    int image = 1;
    for(Mat &latent: feats){
        cout << "image " << image << ": " << latent << endl;
        image++;
    }
    session->Close();
    return 0;
}
