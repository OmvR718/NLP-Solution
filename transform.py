# Smart RAG Document Chunker
import os
import re
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

class SmartRAGChunker:
    """
    Intelligent hierarchical document chunker for RAG systems
    Creates parent and child chunks optimized for retrieval and context understanding
    """
    
    def __init__(self, model_context_window: int = 4096):
        self.sections = {}
        self.hierarchical_chunks = {}
        self.all_chunks = []
        self.model_context_window = model_context_window
        self.source_folder = None  # Track source folder for output
        
        # Smart chunking configuration
        self.config = {
            # Use 70% of context window for content, reserve 30% for prompt/response
            'target_context_usage': 0.70,
            'available_context': int(model_context_window * 0.70),
            
            # Optimal chunk sizes (1 token ≈ 4 characters)
            'parent_chunk_size': 1200,    # ~300 tokens - larger context pieces
            'child_chunk_size': 400,      # ~100 tokens - retrieval-optimized
            'overlap_size': 50,           # Maintains semantic continuity
            
            # Quality settings
            'min_chunk_size': 100,
            'sentence_boundary_preference': True,
            'preserve_code_blocks': True,
        }
        
        # Domain-specific acronyms with explanations
        self.domain_acronyms = {
            'UE': 'User Equipment (mobile device)',
            'MME': 'Mobility Management Entity (core network)',
            'HSS': 'Home Subscriber Server (user database)',
            'eNodeB': 'Evolved Node B (base station)',
            'EPS': 'Evolved Packet System (LTE core)',
            'PDN': 'Packet Data Network (internet)',
            'QCI': 'QoS Class Identifier (service quality)',
            'APN': 'Access Point Name (network gateway)',
            'IMSI': 'International Mobile Subscriber Identity',
            'GUTI': 'Globally Unique Temporary Identifier',
            'VoLTE': 'Voice over LTE (voice calls)',
            'IMS': 'IP Multimedia Subsystem (services)',
            'SIP': 'Session Initiation Protocol (signaling)',
            'RTP': 'Real-time Transport Protocol (media)',
            'PCRF': 'Policy and Charging Rules Function',
            'PGW': 'Packet Data Network Gateway',
            'SGW': 'Serving Gateway (data forwarding)',
            'TAU': 'Tracking Area Update (location)',
            'NAS': 'Non-Access Stratum (signaling)',
            'EUTRA': 'Evolved Universal Terrestrial Radio Access',
            'IoT': 'Internet of Things',
            'PCC': 'Policy And Charging Control',
            'EPC': 'Evolved Packet Core',
            'RLC': 'Radio Link Control',
            'UL': 'Up Link',
            'TFT': 'Traffic Flow Template',
        }

    def load_sections(self, file_pattern: str = "*.txt", folder_path: str = "./") -> bool:
        """Load text files and prepare them for chunking"""
        print("📁 Loading documents for smart chunking...")
        print(f"📊 Target Context Usage: {self.config['target_context_usage']:.0%}")
        print(f"📏 Available Context: {self.config['available_context']} characters")
        
        # Store the source folder for output
        self.source_folder = os.path.abspath(folder_path)
        
        # Find matching files
        pattern = os.path.join(folder_path, file_pattern) if folder_path != "./" else file_pattern
        txt_files = sorted(glob.glob(pattern))
        
        if not txt_files:
            print("❌ No .txt files found!")
            return False
        
        print(f"📁 Found {len(txt_files)} files:")
        
        # Process each file
        for file_path in txt_files:
            section_name = self._clean_section_name(os.path.splitext(os.path.basename(file_path))[0])
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Skip files that are too short
                if len(content) < self.config['min_chunk_size']:
                    print(f"  ⚠️  Skipped '{section_name}': too short ({len(content)} chars)")
                    continue
                
                # Clean and prepare content
                content = self._preprocess_content(content)
                self.sections[section_name] = content
                
                # Show progress
                estimated_tokens = len(content) // 4
                print(f"  ✅ '{section_name}': {len(content)} chars (~{estimated_tokens} tokens)")
                    
            except Exception as e:
                print(f"  ❌ Error loading '{section_name}': {e}")
        
        # Show summary
        if self.sections:
            total_chars = sum(len(content) for content in self.sections.values())
            total_tokens = total_chars // 4
            print(f"\n📊 Successfully loaded {len(self.sections)} sections")
            print(f"📏 Total content: {total_chars:,} chars (~{total_tokens:,} tokens)")
            
            # Estimate retrieval efficiency
            chunks_per_query = self.config['available_context'] // self.config['child_chunk_size']
            print(f"🎯 Estimated {chunks_per_query} child chunks per query")
            
        return len(self.sections) > 0

    def _clean_section_name(self, name: str) -> str:
        """Clean up section names to be consistent"""
        clean_name = re.sub(r'[^\w\s-]', '', name)
        clean_name = re.sub(r'\s+', '_', clean_name.strip())
        return clean_name.lower()

    def _preprocess_content(self, content: str) -> str:
        """Clean and enhance content for better chunking"""
        
        # Temporarily store code blocks to preserve them
        code_blocks = []
        def save_code(match):
            code_blocks.append(match.group(0))
            return f"<<CODE_BLOCK_{len(code_blocks)-1}>>"
        
        content = re.sub(r'```[\s\S]*?```', save_code, content)
        
        # Clean up text formatting
        content = re.sub(r'\r\n', '\n', content)        # Normalize line endings
        content = re.sub(r'\n{3,}', '\n\n', content)    # Remove excessive blank lines
        content = re.sub(r'[ \t]+', ' ', content)       # Normalize spaces
        content = re.sub(r' \n', '\n', content)         # Remove trailing spaces
        
        # Restore code blocks
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f"<<CODE_BLOCK_{i}>>", code_block)
        
        # Add helpful acronym definitions
        content = self._add_acronym_definitions(content)
        
        return content.strip()

    def _add_acronym_definitions(self, content: str) -> str:
        """Add definitions for acronyms to help with understanding"""
        defined_acronyms = set()
        
        for acronym, definition in self.domain_acronyms.items():
            # Look for the acronym as a standalone word
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, content) and acronym not in defined_acronyms:
                # Replace first occurrence with acronym + definition
                replacement = f"{acronym} ({definition})"
                content = re.sub(pattern, replacement, content, count=1)
                defined_acronyms.add(acronym)
        
        return content

    def create_hierarchical_chunks(self) -> List[Dict]:
        """Create smart hierarchical chunks optimized for RAG systems"""
        print(f"\n🏗️ Creating Smart Hierarchical Chunks")
        print(f"📏 Parent chunks: ~{self.config['parent_chunk_size']} chars (broader context)")
        print(f"📏 Child chunks: ~{self.config['child_chunk_size']} chars (retrieval-optimized)")
        print(f"🔗 Overlap: {self.config['overlap_size']} chars (semantic continuity)")
        
        self.hierarchical_chunks = {}
        self.all_chunks = []
        global_id = 1
        
        for section_name, content in self.sections.items():
            print(f"\n--- Processing: {section_name} ---")
            
            # Create parent chunks (larger context pieces)
            parent_chunks = self._create_parent_chunks(section_name, content)
            print(f"  📦 Created {len(parent_chunks)} parent chunks")
            
            section_data = {
                'section_name': section_name,
                'original_length': len(content),
                'parent_chunks': parent_chunks,
                'child_chunks': []
            }
            
            # Create child chunks from each parent (for retrieval)
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                child_chunks = self._create_child_chunks(
                    section_name, parent_chunk, parent_idx
                )
                section_data['child_chunks'].extend(child_chunks)
                parent_chunk['child_chunk_ids'] = [c['chunk_id'] for c in child_chunks]
                parent_chunk['child_count'] = len(child_chunks)
            
            print(f"  📄 Created {len(section_data['child_chunks'])} child chunks")
            
            # Assign global IDs and store all chunks
            for chunk in parent_chunks + section_data['child_chunks']:
                chunk['global_id'] = global_id
                self.all_chunks.append(chunk)
                global_id += 1
            
            self.hierarchical_chunks[section_name] = section_data
        
        # Show optimization metrics
        self._calculate_optimization_metrics()
        return self.all_chunks

    def _create_parent_chunks(self, section_name: str, content: str) -> List[Dict]:
        """Create parent chunks with intelligent sizing"""
        parent_chunks = []
        chunk_size = self.config['parent_chunk_size']
        overlap = self.config['overlap_size']
        
        if len(content) <= chunk_size:
            # Content fits in a single parent chunk
            chunk = {
                'chunk_id': f"{section_name}_P1",
                'level': 'parent',
                'section_name': section_name,
                'chunk_index': 1,
                'content': content,
                'char_count': len(content),
                'estimated_tokens': len(content) // 4,
                'content_hash': hashlib.md5(content.encode()).hexdigest()[:8]
            }
            parent_chunks.append(chunk)
        else:
            # Split content into multiple parent chunks
            chunks = self._smart_split_content(content, chunk_size, overlap)
            for i, chunk_content in enumerate(chunks, 1):
                chunk = {
                    'chunk_id': f"{section_name}_P{i}",
                    'level': 'parent',
                    'section_name': section_name,
                    'chunk_index': i,
                    'content': chunk_content,
                    'char_count': len(chunk_content),
                    'estimated_tokens': len(chunk_content) // 4,
                    'content_hash': hashlib.md5(chunk_content.encode()).hexdigest()[:8]
                }
                parent_chunks.append(chunk)
        
        return parent_chunks

    def _create_child_chunks(self, section_name: str, parent_chunk: Dict, parent_idx: int) -> List[Dict]:
        """Create child chunks optimized for retrieval"""
        child_chunks = []
        chunk_size = self.config['child_chunk_size']
        overlap = self.config['overlap_size']
        content = parent_chunk['content']
        
        if len(content) <= chunk_size:
            # Parent content fits in a single child chunk
            chunk = {
                'chunk_id': f"{section_name}_P{parent_idx+1}_C1",
                'level': 'child',
                'section_name': section_name,
                'parent_id': parent_chunk['chunk_id'],
                'chunk_index': 1,
                'content': content,
                'char_count': len(content),
                'estimated_tokens': len(content) // 4,
                'content_hash': hashlib.md5(content.encode()).hexdigest()[:8]
            }
            child_chunks.append(chunk)
        else:
            # Split parent content into multiple child chunks
            chunks = self._smart_split_content(content, chunk_size, overlap)
            for i, chunk_content in enumerate(chunks, 1):
                chunk = {
                    'chunk_id': f"{section_name}_P{parent_idx+1}_C{i}",
                    'level': 'child',
                    'section_name': section_name,
                    'parent_id': parent_chunk['chunk_id'],
                    'chunk_index': i,
                    'content': chunk_content,
                    'char_count': len(chunk_content),
                    'estimated_tokens': len(chunk_content) // 4,
                    'content_hash': hashlib.md5(content.encode()).hexdigest()[:8]
                }
                child_chunks.append(chunk)
        
        return child_chunks

    def _smart_split_content(self, content: str, target_size: int, overlap: int) -> List[str]:
        """Intelligently split content while respecting natural boundaries"""
        chunks = []
        
        # Start by splitting on paragraphs (most natural boundary)
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # Try adding the whole paragraph
            test_chunk = current_chunk + '\n\n' + para if current_chunk else para
            
            if len(test_chunk) <= target_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if we have one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Create overlap for context continuity
                    if overlap > 0 and len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        # Find a good sentence boundary for clean overlap
                        sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
                        current_chunk = sentences[-1] if sentences else overlap_text
                    else:
                        current_chunk = ""
                
                # Handle paragraphs that are too long
                if len(para) > target_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    temp_chunk = current_chunk
                    
                    for sentence in sentences:
                        if len(temp_chunk + ' ' + sentence) <= target_size:
                            temp_chunk = temp_chunk + ' ' + sentence if temp_chunk else sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                # Handle overlap
                                if overlap > 0:
                                    overlap_part = temp_chunk[-overlap:] if len(temp_chunk) > overlap else temp_chunk
                                    temp_chunk = overlap_part + ' ' + sentence
                                else:
                                    temp_chunk = sentence
                            else:
                                # Even single sentence is too long - just add it
                                chunks.append(sentence.strip())
                                temp_chunk = ""
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _calculate_optimization_metrics(self):
        """Calculate and display chunking optimization metrics"""
        parent_chunks = [c for c in self.all_chunks if c['level'] == 'parent']
        child_chunks = [c for c in self.all_chunks if c['level'] == 'child']
        
        # Calculate token statistics
        parent_tokens = [c['estimated_tokens'] for c in parent_chunks]
        child_tokens = [c['estimated_tokens'] for c in child_chunks]
        
        # Analyze context window utilization
        avg_child_tokens = sum(child_tokens) / len(child_tokens) if child_tokens else 0
        max_chunks_per_query = self.config['available_context'] // (avg_child_tokens * 4)
        
        print(f"\n📊 Chunking Optimization Metrics:")
        print(f"   📦 Parent chunks: {len(parent_chunks)} (avg {sum(parent_tokens)/len(parent_tokens):.0f} tokens)")
        print(f"   📄 Child chunks: {len(child_chunks)} (avg {avg_child_tokens:.0f} tokens)")
        print(f"   🎯 Optimal chunks per query: {max_chunks_per_query:.0f}")
        print(f"   📈 Context Window Usage: {(max_chunks_per_query * avg_child_tokens * 4 / self.model_context_window):.1%}")

    def save_chunked_output(self, base_filename: str = "rag_chunks") -> Dict[str, str]:
        """Save chunks in multiple formats to the output folder"""
        
        if not self.all_chunks:
            print("❌ No chunks to save!")
            return {}
        
        # Create output folder in the same directory as the source files
        if self.source_folder:
            output_folder = os.path.join(self.source_folder, 'output')
        else:
            output_folder = os.path.join(os.getcwd(), 'output')
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Created output folder: {output_folder}")
        
        files_created = {}
        
        # 1. Human-readable hierarchical format
        hierarchical_file = os.path.join(output_folder, f"{base_filename}_hierarchical.txt")
        self._save_hierarchical_format(hierarchical_file)
        files_created['hierarchical'] = hierarchical_file
        
        # 2. Structured JSON format
        json_file = os.path.join(output_folder, f"{base_filename}_structured.json")
        self._save_json_format(json_file)
        files_created['json'] = json_file
        
        # 3. RAG-ready format for vector databases
        rag_file = os.path.join(output_folder, f"{base_filename}_rag_ready.txt")
        self._save_rag_format(rag_file)
        files_created['rag'] = rag_file
        
        # 4. Metadata and statistics
        metadata_file = os.path.join(output_folder, f"{base_filename}_metadata.json")
        self._save_metadata(metadata_file)
        files_created['metadata'] = metadata_file
        
        print(f"\n💾 Created {len(files_created)} output files in the 'output' folder:")
        for purpose, filename in files_created.items():
            print(f"   📄 {purpose}: {os.path.basename(filename)}")
        
        return files_created

    def _save_hierarchical_format(self, filename: str):
        """Save human-readable hierarchical format"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SMART RAG HIERARCHICAL CHUNKS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model Context Window: {self.model_context_window} tokens\n")
            f.write(f"Target Context Usage: {self.config['target_context_usage']:.0%}\n")
            f.write(f"Structure: Section → Parent Chunks → Child Chunks\n")
            f.write("=" * 80 + "\n\n")
            
            for section_name, section_data in self.hierarchical_chunks.items():
                f.write(f"{'#' * 80}\n")
                f.write(f"# SECTION: {section_name.upper()}\n")
                f.write(f"# Original: {section_data['original_length']:,} chars\n")
                f.write(f"# Parents: {len(section_data['parent_chunks'])}, Children: {len(section_data['child_chunks'])}\n")
                f.write(f"{'#' * 80}\n\n")
                
                # Write parent chunks
                for parent in section_data['parent_chunks']:
                    f.write(f"📦 PARENT CHUNK: {parent['chunk_id']}\n")
                    f.write(f"   Tokens: ~{parent['estimated_tokens']}, Children: {parent['child_count']}\n")
                    f.write(f"   Hash: {parent['content_hash']}\n")
                    f.write(f"   {'-' * 60}\n")
                    f.write(f"   {parent['content'][:200]}...\n")
                    f.write(f"   {'-' * 60}\n\n")
                    
                    # Write corresponding child chunks
                    child_chunks = [c for c in section_data['child_chunks'] if c['parent_id'] == parent['chunk_id']]
                    for child in child_chunks:
                        f.write(f"   📄 CHILD: {child['chunk_id']}\n")
                        f.write(f"      Tokens: ~{child['estimated_tokens']}, Hash: {child['content_hash']}\n")
                        f.write(f"      {'-' * 40}\n")
                        f.write(f"      {child['content']}\n")
                        f.write(f"      {'-' * 40}\n\n")
                
                f.write(f"{'=' * 80}\n\n")

    def _save_json_format(self, filename: str):
        """Save structured JSON format for programmatic use"""
        data = {
            'metadata': {
                'model_context_window': self.model_context_window,
                'target_context_usage': self.config['target_context_usage'],
                'chunk_sizes': {
                    'parent': self.config['parent_chunk_size'],
                    'child': self.config['child_chunk_size'],
                    'overlap': self.config['overlap_size']
                },
                'total_sections': len(self.hierarchical_chunks),
                'total_chunks': len(self.all_chunks)
            },
            'sections': self.hierarchical_chunks
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_rag_format(self, filename: str):
        """Save RAG-ready format for embedding and vector databases"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# SMART RAG-READY CHUNKS\n")
            f.write("# Format: ID|LEVEL|SECTION|PARENT|TOKENS|HASH|CONTENT\n\n")
        
            for chunk in self.all_chunks:
                parent_id = chunk.get('parent_id', 'ROOT')
                # Clean content for safe storage
                safe_content = (
                    chunk['content']
                    .replace('|', '&#124;')
                    .replace('\n', '\\n')
                )
                line = (
                    f"{chunk['chunk_id']}|{chunk['level']}|{chunk['section_name']}|"
                    f"{parent_id}|{chunk['estimated_tokens']}|{chunk['content_hash']}|"
                    f"{safe_content}\n"
                )
                f.write(line)

    def _save_metadata(self, filename: str):
        """Save metadata and statistics"""
        parent_chunks = [c for c in self.all_chunks if c['level'] == 'parent']
        child_chunks = [c for c in self.all_chunks if c['level'] == 'child']
        
        metadata = {
            'chunking_config': {
                'model_context_window': self.model_context_window,
                'target_context_usage': self.config['target_context_usage'],
                'available_context': self.config['available_context']
            },
            'chunk_statistics': {
                'total_sections': len(self.sections),
                'parent_chunks': len(parent_chunks),
                'child_chunks': len(child_chunks),
                'total_chunks': len(self.all_chunks),
                'avg_parent_tokens': sum(c['estimated_tokens'] for c in parent_chunks) / len(parent_chunks),
                'avg_child_tokens': sum(c['estimated_tokens'] for c in child_chunks) / len(child_chunks) if child_chunks else 0
            },
            'section_breakdown': {
                section: {
                    'original_chars': data['original_length'],
                    'parent_count': len(data['parent_chunks']),
                    'child_count': len(data['child_chunks'])
                }
                for section, data in self.hierarchical_chunks.items()
            },
            'recommendations': self._get_optimization_recommendations()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _get_optimization_recommendations(self) -> Dict[str, str]:
        """Generate helpful optimization recommendations"""
        child_chunks = [c for c in self.all_chunks if c['level'] == 'child']
        avg_child_tokens = sum(c['estimated_tokens'] for c in child_chunks) / len(child_chunks) if child_chunks else 0
        
        recommendations = {}
        
        # Context utilization recommendation
        optimal_chunks = self.config['available_context'] // (avg_child_tokens * 4)
        if optimal_chunks < 3:
            recommendations['chunk_size'] = "Consider smaller child chunks for better context utilization"
        elif optimal_chunks > 8:
            recommendations['chunk_size'] = "Consider larger child chunks for efficiency"
        else:
            recommendations['chunk_size'] = f"Optimal: ~{optimal_chunks:.0f} chunks per query"
        
        # Retrieval strategy recommendation
        if len(child_chunks) > 100:
            recommendations['retrieval'] = "Use semantic similarity + parent-child relationships for retrieval"
        else:
            recommendations['retrieval'] = "Simple top-k retrieval should work well"
        
        return recommendations


# Main usage functions

def process_documents(folder_path: str = "./", context_window: int = 4096) -> SmartRAGChunker:
    """Main function to process documents for RAG systems"""
    print("🚀 Smart RAG Document Processing")
    print("=" * 50)
    
    # Initialize the chunker
    chunker = SmartRAGChunker(model_context_window=context_window)
    
    # Load and process documents
    if not chunker.load_sections("*.txt", folder_path):
        return None
    
    # Create hierarchical chunks
    chunks = chunker.create_hierarchical_chunks()
    
    # Save in multiple formats to the output folder
    files = chunker.save_chunked_output("smart_rag_chunks")
    
    print(f"\n🎯 Ready for RAG system integration!")
    print(f"📁 All files saved to 'output' folder")
    
    return chunker

def quick_process():
    """Quick processing for current directory with standard settings"""
    return process_documents("./", 4096)

def process_large_context():
    """Processing for models with large context windows (e.g., 128K tokens)"""
    return process_documents("./", context_window=128000)

# Example usage
if __name__ == "__main__":
    print("🚀 Running Smart RAG Document Chunker...\n")

    # Process documents in the specified folder
    chunker = process_documents("f:/RAG/", 4096)

    if chunker:
        print("\n✅ Processing complete! Output files saved to 'output' folder:")
        print(" - smart_rag_chunks_hierarchical.txt (human-readable)")
        print(" - smart_rag_chunks_structured.json (programmatic)")
        print(" - smart_rag_chunks_rag_ready.txt (vector database)")
        print(" - smart_rag_chunks_metadata.json (statistics)")
    else:
        print("❌ Processing failed - check your input files")