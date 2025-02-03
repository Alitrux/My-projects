#include "ConfusionMatrix.hpp"
#include <iostream>

using namespace std;

ConfusionMatrix::ConfusionMatrix() {
    int m_confusion_matrix[2][2];
    m_confusion_matrix[0][0] = 0;
    m_confusion_matrix[0][1] = 0;
    m_confusion_matrix[1][0] = 0;
    m_confusion_matrix[1][1] = 0;

}

ConfusionMatrix::~ConfusionMatrix() {
    // Attribute m_confusion_matrix is deleted automatically
}

void ConfusionMatrix::add_prediction(int true_label, int predicted_label) {
    ++m_confusion_matrix[true_label][predicted_label];
}

void ConfusionMatrix::print_evaluation() const{
    // Prints the confusion matrix
    cout << "\t\tPredicted\n";
    cout << "\t\t0\t1\n";
    cout << "Actual\t0\t"
        << get_tn() <<"\t"
        << get_fp() <<endl;
    cout <<"\t1\t"
        << get_fn() <<"\t"
        << get_tp() <<endl <<endl;
    // Prints the estimators
    cout << "Error rate\t\t"
        << error_rate() <<endl;
    cout << "False alarm rate\t"
        << false_alarm_rate() <<endl;
    cout << "Detection rate\t\t"
        << detection_rate() <<endl;
    cout << "F-score\t\t\t"
        << f_score() <<endl;
    cout << "Precision\t\t"
        << precision() <<endl;
}

int ConfusionMatrix::get_tp() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix::get_tn() const {
   return m_confusion_matrix[0][0];
}

int ConfusionMatrix::get_fp() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix::get_fn() const {
   return m_confusion_matrix[1][0];
}

double ConfusionMatrix::f_score() const {
    return 2 * precision() * detection_rate() / (precision() + detection_rate());
}

double ConfusionMatrix::precision() const {
    return static_cast<double>(get_tp()) / (static_cast<double>(get_fp()) + static_cast<double>(get_tp()));
}

double ConfusionMatrix::error_rate() const {
    double tot = static_cast<double>(get_tp() + get_tn() + get_fp() + get_fn());
    return (static_cast<double>(get_fp() + get_fn())) / tot;
}

double ConfusionMatrix::detection_rate() const {
    return static_cast<double>(get_tp()) / static_cast<double>(get_tp() + get_fn());
}

double ConfusionMatrix::false_alarm_rate() const {
    return static_cast<double>(get_fp()) / static_cast<double>(get_fp() + get_tn());
}
