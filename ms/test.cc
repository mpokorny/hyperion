#include <experimental/filesystem>
#include <map>
#include <memory>
#include <vector>

#include "legion.h"

#include "IndexTree.h"
#include "Column.h"
#include "SpectralWindowTable.h"
#include "Table.h"
#include "TableReadTask.h"

using namespace legms;
using namespace legms::ms;
using namespace Legion;

enum {
  TOP_LEVEL_TASK_ID,
};

class TopLevelTask
{
public:

  static constexpr const char *TASK_NAME = "top_level";
  static const int TASK_ID = TOP_LEVEL_TASK_ID;

  static void
  base_impl(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime* runtime) {

    TableReadTask::register_task(runtime);
    TreeIndexSpace::register_tasks(runtime);
    FillProjectionsTasks::register_tasks(runtime);

    std::experimental::filesystem::path ms_path = "foo.ms";
    SpectralWindowTable spectral_window_table(ms_path);
    std::cout << "name: "
              << spectral_window_table.name() << std::endl;
    std::cout << "columns: ";
    std::vector<std::string> colnames =
      {"NUM_CHAN", "TOTAL_BANDWIDTH", "CHAN_FREQ", "ASSOC_SPW_ID"};
    std::for_each (
      colnames.begin(),
      colnames.end(),
      [](auto& nm) { std::cout << nm << " "; });
    std::cout << std::endl;

    TableReadTask spectral_window_read_task(
      spectral_window_table.path(),
      spectral_window_table,
      colnames);
    auto lr_fids = spectral_window_read_task.dispatch(ctx, runtime);
    assert(lr_fids.size() == colnames.size());
    {
      auto& [lr, fid] = lr_fids[0];
      auto launcher = InlineLauncher(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(fid);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      const FieldAccessor<READ_ONLY, int, 1> num_chans(pr, fid);
      DomainT<1> domain = runtime->get_index_space_domain(lr.get_index_space());
      std::cout << "num_chan: ";
      for (PointInDomainIterator<1> pid(domain); pid(); pid++)
        std::cout << num_chans[*pid] << " ";
      std::cout << std::endl;
    }
    {
      auto& [lr, fid] = lr_fids[1];
      auto launcher = InlineLauncher(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(fid);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      const FieldAccessor<READ_ONLY, double, 1> total_bws(pr, fid);
      DomainT<1> domain = runtime->get_index_space_domain(lr.get_index_space());
      std::cout << "total bw: ";
      for (PointInDomainIterator<1> pid(domain); pid(); pid++)
        std::cout << total_bws[*pid] << " ";
      std::cout << std::endl;
    }
    {
      auto& [lr, fid] = lr_fids[2];
      auto launcher = InlineLauncher(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(fid);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      const FieldAccessor<READ_ONLY, double, 2> chan_freqs(pr, fid);
      DomainT<2> domain = runtime->get_index_space_domain(lr.get_index_space());
      std::cout << "chan freqs: ";
      std::optional<coord_t> row_index;
      for (PointInDomainIterator<2> pid(domain); pid(); pid++) {
        if (!row_index) {
          std::cout << "(";
          row_index = pid[0];
        }
        else if (row_index.value() != pid[0]) {
          std::cout << "),(";
          row_index = pid[0];
        }
        std::cout << chan_freqs[*pid] << ",";
      }
      std::cout << ")" << std::endl;
    }
    {
      auto& [lr, fid] = lr_fids[3];
      auto launcher = InlineLauncher(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(fid);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      const FieldAccessor<
        READ_ONLY,
        std::vector<int>,
        1,
        coord_t,
        Realm::AffineAccessor<std::vector<int>,1,coord_t>,
        false> assoc_spws(pr, fid);
      DomainT<1> domain = runtime->get_index_space_domain(lr.get_index_space());
      std::cout << "assoc spws: ";
      for (PointInDomainIterator<1> pid(domain); pid(); pid++) {
        std::cout << "(";
        auto spws = assoc_spws[*pid];
        for (auto& spw : spws)
          std::cout << spw << ",";
        std::cout << "),";
      }
      std::cout << std::endl;
    }
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }
};

int
main(int argc, char** argv) {

  Runtime::set_top_level_task_id(TopLevelTask::TASK_ID);
  TopLevelTask::register_task();
  SerdezManager::register_ops();
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:

