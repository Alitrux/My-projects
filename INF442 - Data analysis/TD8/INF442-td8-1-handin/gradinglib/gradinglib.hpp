#ifndef TESTLIB_HPP
#define TESTLIB_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>

namespace testlib {

  bool match(const std::string &str, const std::string regexp);

  void print(std::ostream &out);

  template <typename T> void print(std::ostream &out, const T& t) {
    out << t;
  }

  template <typename First, typename... Rest>
  void print(std::ostream &out, const First& first, const Rest&... rest) {
    out << first << ", ";
    print(out, rest...); // recursive call using pack expansion syntax
  }

  template<typename _CharT, typename _Traits>
  std::basic_ostream<_CharT, _Traits>&
  linesep(std::basic_ostream<_CharT, _Traits>& __os) {
    char buffer[80];
    for (auto i=0; i < 80; ++i) buffer[i] = '-';
    return (__os.write(buffer,80));
  }

  template <typename T>
  std::string get_vector_str(const std::string tname, std::vector<T> vec) {
    typedef typename std::vector<T>::const_iterator titer;
    titer it;
    std::stringstream s;
//    s << "std::vector<" << tname << ">{";
    s << "[";
    for (it = vec.begin(); it != vec.end(); ++it) {
      if (it != vec.begin())
        s << ", ";
      s << *it;
    }
    s << "]";
    return s.str();
  };

  template <typename T>
  std::string get_array_str(T* ptr, const int size) {
    std::stringstream s;
    s << "[";
    for (int i = 0; i < size; i++) {
      if (i > 0)
        s << ", ";
      s << *(ptr+i);
    }
    s << "]";
    return s.str();
  };


  template <typename... Arguments>
  void print_tested_function(std::ostream &out,
                             const std::string &function_name,
    			     const Arguments&... args) {
    // not printing arguments by default
    out << function_name;
  }

  template <typename... Arguments>
  void print_tested_function_input(std::ostream &out,
                                   const std::string &function_name,
                                   const Arguments&... args) {
    out << function_name << " called with inputs: ";
    print(out, args...);
  }

  std::string get_success();

  std::string get_failure();

  template <typename T>
  void print_array(std::ostream &out, T* array, int length) {
    out << "[";
    for(int i = 0; i < length; ++i) {
      if (i != 0) out << ", ";
      out << array[i];
    }
    out << "]";
  }

  template <typename T1, typename T2, typename... Arguments>
  bool test_eq(std::ostream &out,
               const std::string &function_name,
               T1 result,
               T2 expected,
               const Arguments&... args) {

    bool success = result == expected;
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << result
        << " expected " << expected;
    out << std::endl;
    return success;
  }

  template <typename... Arguments>
  bool test_eq(std::ostream &out,
               const std::string &function_name,
               std::vector<int> result,
               std::vector<int> expected,
               const Arguments&... args) {

    bool success = result == expected;
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    if (not success) {
      auto mypair = std::mismatch(result.begin(), result.end(),
        expected.begin());
      out << "\nFirst mismatch at " << mypair.first - result.begin() << "; expected: " << *mypair.first << " got: " << *mypair.second;
    }
    out << std::endl;
    return success;
  }

  template <typename... Arguments>
  bool test_eq(std::ostream &out,
               const std::string &function_name,
               std::vector<std::vector<double>> result,
               std::vector<std::vector<double>> expected,
               const Arguments&... args) {

    bool success = result == expected;
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);
    if (not success) {
      auto mypair = std::mismatch(result.begin(), result.end(),
        expected.begin());
      std::ostringstream first;
      std::ostringstream second;
      if (!(*mypair.first).empty()) {
        std::copy((*mypair.first).begin(), (*mypair.first).end() - 1,
          std::ostream_iterator<double>(first, ", "));
        first << (*mypair.first).back();
      }
      if (!(*mypair.first).empty()) {
        std::copy((*mypair.second).begin(), (*mypair.second).end() - 1,
          std::ostream_iterator<double>(second, ", "));
        second << (*mypair.second).back();
      }
      out << "\nFirst mismatch at row " << (mypair.first - result.begin()) << "; expected: " << first.str() << " got: " << second.str();
    }
    out << std::endl;
    return success;
  }

  template <typename T1, typename T2, typename... Arguments>
  bool test_le(std::ostream &out,
               const std::string &function_name,
               T1 result,
               T2 expected,
               const Arguments&... args) {

    bool success = result <= expected;
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << result
        << " expected less or equal than " << expected;
    out << std::endl;

    return success;
  }


  template <typename T>
  void print_ptr_value(std::ostream &out,
                       T* ptr) {
    if (ptr != NULL)
      out << *ptr;
    else
      out << "NULL";
  }

  template <typename T1, typename T2, typename... Arguments>
  bool test_eq_ptr(std::ostream &out,
                   const std::string &function_name,
                   T1* result,
                   T2* expected,
                   const Arguments&... args) {

    bool success = result == expected;
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got ";
    print_ptr_value(out, result);
    out << " (element at address " << result << ")"
        << " expected ";
    print_ptr_value(out, expected);
    out << " (element at address " << expected << ")";
    out << std::endl;

    return success;
  }


  template <typename T, typename... Arguments>
  bool test_eq_approx(std::ostream &out,
                      const std::string &function_name,
                      T result,
                      T expected,
                      T delta,
                      const Arguments&... args) {
    bool success;
    // May not work on some types.
    if (result > expected)
      success = result <= (expected + delta);
    else
      success = expected <= (result + delta);

    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << result
        << " expected " << expected << " + [-" << delta << "," << delta << "]";
    out << std::endl;

    return success;
  }

  template <typename... Arguments>
  bool test_eq_approx(std::ostream &out,
                      const std::string &function_name,
                      std::vector<std::vector<double>> result,
                      std::vector<std::vector<double>> expected,
                      double delta,
                      const Arguments&... args) {
    bool success;
    // May not work on some types.
    success = result.size() == expected.size();
    for (std::size_t i=0; i < result.size(); i++) {
      success = success && (result[i].size() == expected[i].size());
      if (!success)
        break;
      for (std::size_t j=0; j < result[i].size(); j++) {
        if (result[i][j] > expected[i][j])
          success = success && (result[i][j] <= (expected[i][j] + delta));
        else
          success = success && (expected[i][j] <= (result[i][j] + delta));
      }
    }
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << "Vector of vectors: either one of the sizes doesn't match, or one element";
    out << std::endl;

    return success;
  }

  template <typename T, typename... Arguments>
  bool test_eq_range(std::ostream &out,
                     const std::string &function_name,
                     T value,
                     T lower_bound,
                     T upper_bound,
                     const Arguments&... args) {
    bool success;

    // May not work on some types.
    success = (value >= lower_bound && value <= upper_bound);

    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << value
        << " expected values in the range [" << lower_bound << "," << upper_bound << "]";
    out << std::endl;

    return success;
  }


  template <typename... Arguments>
  bool test_in_output(std::ostream &out, std::string& func_name,
                      std::string func (const char* in),
                      const char* input,
                      std::vector<std::string> &expected) {
    std::vector<bool> test_res;
    std::string prog_out = func(input);
    bool success = true;
    for (auto str : expected) {
      if (match(prog_out, str))
        test_res.push_back(true);
      else {
        success = false;
        test_res.push_back(false);
      }
    }
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function_input(out, func_name, input);
    out << std::endl;

    const std::string filler = "          ";
    auto iter_res = test_res.begin();
    for(auto iter_ex = expected.begin(); expected.end() != iter_ex; ++iter_ex, ++iter_res) {
      if (*iter_res)
        out << filler << "Correctly found this string in the output: \"" << *iter_ex << "\"" << std::endl;
      else
        out << filler << "We did not find this string in the output: \"" << *iter_ex << "\"" << std::endl;
    }

    if (! success) {
      out << filler << "Your implementation returned the following (wrong) output: " << std::endl
          << linesep << std::endl
          << prog_out << std::endl
          << linesep << std::endl;
    }

    return success;
  }


  void start_test_suite(std::ostream &out, const std::string &name);

  int end_test_suite(std::ostream &out, const std::string &name,
                     const int &correct, const int &total);

  int compute_final_score(std::ostream &out,
                          const std::vector<std::pair<int,int>> scores,
                          const std::string test_names[]);

  int run_grading(std::ostream &out, const int test_case_number, int const total_test_cases,
                  std::string const test_names[],
                  int const points[],
                  int (*test_functions[]) (std::ostream &, const std::string));
}

#endif // TESTLIB_HPP
