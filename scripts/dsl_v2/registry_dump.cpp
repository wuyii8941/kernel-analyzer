// W0 (rc3 04): enumerate the operations a built triton-opt registers.  The same dialect registration as
// bin/triton-opt.cpp (registerTritonDialects), then MLIRContext::getRegisteredOperations() after loading every
// available dialect.  Built with triton-opt's own compile / link commands (scripts/dsl_v2/build_registry_dump.sh),
// so the registry is the one of that build.  Output: one "dialect<TAB>operation" line per registered op.
#include "./RegisterTritonDialects.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (mlir::RegisteredOperationName op : context.getRegisteredOperations())
    llvm::outs() << op.getDialectNamespace() << "\t" << op.getStringRef() << "\n";
  for (mlir::Dialect *d : context.getLoadedDialects())
    llvm::errs() << "dialect\t" << d->getNamespace() << "\n";
  return 0;
}
